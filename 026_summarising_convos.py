# the longer you continue a convo, the costlier it will be to maintain context. It can get expensive to store a large amount of chat history. Hence, what we do is, after each exchange of messages, we summarise the convo so far using a module called ConversationSummaryMemory. 
# a sub-chain within the memory object takes in the chat messages, and summarises them. This summary is then passed on as a system message during a follow up chat.

# note: ConversationSummaryMemory does not work well with filechatmessagehistory apparently. dont' use them together.

from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import HumanMessagePromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationSummaryMemory
from dotenv import load_dotenv

load_dotenv()

chat = ChatOpenAI(verbose = True) 

# we add the 'llm' arg to memory because it now runs a subchain (for summarising text), which needs some llm to use.
memory = ConversationSummaryMemory(memory_key="messages_list",return_messages=True, llm = chat)

prompt = ChatPromptTemplate(
    input_variables=["message_content", "messages_list"],
    messages=[
        MessagesPlaceholder(variable_name = "messages_list"),
        HumanMessagePromptTemplate.from_template("{message_content}")
    ]
)

chain = LLMChain(
    llm = chat,
    prompt = prompt,
    memory = memory,
    verbose = True # basically shows steps
)

while True:
    content = input(">> ")
    result = chain.invoke({"message_content": content})
    print(result["text"])
