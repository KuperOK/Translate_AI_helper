import streamlit as st
import openai
import io
import os
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from datetime import datetime
from stqdm import stqdm

def check_openai_api_key(api_key):
    
    """
    Validates the provided OpenAI API key by attempting to list the available models.

    Args:
        api_key (str): The OpenAI API key to be validated.

    Returns:
        bool: True if the API key is valid and the API request succeeds, False otherwise.
    """
    
    client = openai.OpenAI(api_key=api_key)
    try:
        client.models.list()
    except openai.AuthenticationError:
        return False
    else:
        return True

def process_text_file(file, num_parts):
    """
    Splits the content of a text file into a specified number of approximately equal parts.
    Args:
        file (io.TextIOWrapper): A file-like object containing text data to be processed.
        num_parts (int): The number of parts to split the text into.
    Returns:
        List[str]: A list of strings, each representing a part of the original text file.    
    Notes:
        - The function reads the content of the file, splits it into roughly equal parts based on the specified number of parts.
        - The text is split at line breaks, and any leftover lines are distributed among the parts.
        - Each part is returned as a single string within a list.
    """

    text = file.read().decode("utf-8")        
    texts_list = text.splitlines()
    avg_length = len(texts_list) // num_parts
    remainder = len(texts_list) % num_parts    
    result = []
    start = 0    
    for i in range(num_parts):
        end = start + avg_length + (1 if i < remainder else 0)
        result.append(texts_list[start:end])
        start = end
    splited_file_to_list_of_strings = ['\n'.join(part) for part in result]    

    return splited_file_to_list_of_strings
    
def get_gpt_response(chat, split_texts):
    """
    Processes a list of text segments through a GPT model and concatenates the responses.

    This function iterates through each text segment in `split_texts`, formats the segment into a prompt 
    according to predefined rules, sends it to the provided GPT model instance via the `chat` object, 
    and collects the model's responses. The responses are concatenated into a single string with each 
    response separated by a newline. Additionally, the time taken for the translation process is displayed.

    Args:
        chat (object): An instance of a chat interface with the GPT model. It should have an `invoke` method 
                       that takes formatted messages and returns a response object with a `content` attribute.
        split_texts (list of str): A list where each element is a text segment to be processed by the GPT model.

    Returns:
        str: A single string containing all translated text segments concatenated, with each segment's response 
              separated by a newline.

    Example:
        >>> chat = SomeChatInterface()
        >>> texts = ["Hello, how are you?", "I am fine, thank you!"]
        >>> translated_text = get_gpt_response(chat, texts)
        >>> print(translated_text)
        "Hi, how are you?\nI'm good, thanks!"
    """    
    translated = ''
    s = datetime.now()
    for part in stqdm(split_texts):        
        customer_text = f"{part}"
        customer_messages = prompt_template.format_messages(                    
                    rules=customer_rules,
                    text=customer_text)
        response = (chat.invoke(customer_messages)).content        
        response = response.replace("```\n", "")
        response = response.replace("\n```", "")
        translated += response + '\n'         
    e = datetime.now()
    st.write(f'Translation time: {e-s}')
    return translated
    

template_string = """You experienced translator. \
Please translate from Germany to language which specified in parenthesis in each line
of the text that is delimited by triple backticks \
using such rules:\n {rules}. \
\ntext: \n```
{text}
```
"""

prompt_template = ChatPromptTemplate.from_template(template_string)

customer_rules = """
If in text line you see (fr) than next text you must translate to French,
if (it) than you must translate to Italian. 
You must translate each line of text which follows after parenthesis. 
For example, for such line: ```view.elements.choice_of_payment_type.payment.type.cost.0=(fr)(Versandkosten``` you fist part of text line leave as is 
```view.elements.choice_of_payment_type.payment.type.cost.0=(fr)``` 
and translate second part ```(Versandkosten```  which follows after ```(fr)```  . \  
If in second part in some word is special symblols such as `&` or `(` or others or in second part is html-tag or hyperlink 
you dont translate that part of text and leave it as is. \
Dont change any special simbols to another. Dont change `)` to `>`.
If in text some words that you can not understand as Garman you leave it as is. \
Translate any words or strings that are part of technical or placeholder formats as well. \
Use official translation style.
Show only translation and dont add any your coments"""


def main():
    
    st.title("Special Translate AI Helper")
    
    # if 'api_key' not in st.session_state:
    #     st.session_state.api_key = ""
    
    # api_key = st.text_input("Enter your OpenAI API key:", type="password")

    llm_model = st.selectbox(
                "Select LLM",
                ("gpt-4o-mini-2024-07-18", "chatgpt-4o-latest")
        )
    num_parts = st.slider('Num parts to split file', min_value=1, max_value=10, value=1, step=1)
    
     
    os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']
    openai.api_key = os.getenv('OPENAI_API_KEY')
    
    if 'translated_text' not in st.session_state:
        st.session_state.translated_text = None
            
    uploaded_file = st.file_uploader("Upload a text file", type="txt")

    if openai.api_key:        
        if uploaded_file:
            st.write('File Seccessfully upload')
                
            with st.spinner(text="Splitting in progress..."):
                processed_text = process_text_file(uploaded_file, num_parts=num_parts)
            st.success("Splitting done!")
                            
            chat = ChatOpenAI(model_name=llm_model, temperature=0)
            if st.session_state.translated_text is None:
                with st.spinner(text="AI in progress ..."):
                    st.session_state.translated_text = get_gpt_response(chat, processed_text)
          
                buffer = io.BytesIO()
                buffer.write(st.session_state.translated_text.encode("utf-8"))
                buffer.seek(0)
                                                
                filename = 'output ' + datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + '.txt'    
                st.download_button(
                            label="Download File",
                            data=buffer,
                            file_name=filename,
                            mime="text/plain"
                        )

if __name__ == "__main__":
    main()
