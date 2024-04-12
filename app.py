import streamlit as st
from langchain.llms import CTransformers
from langchain.prompts import PromptTemplate


## Function to get response form the LLama2 Model
def getLLamaResponse(input_text, no_of_words, blog_style):
    llm = CTransformers(model='/Users/karthiksagar/BlogGeneration/llama-2-7b-chat.bin',
                        model_type='llama',
                        config={"max_new_tokens": 256,
                                'temperature': 0.01})
    template = """
        write a blog for {blog_style} job profile for a topic {input_text}
        within {no_of_words} words.
            """
    prompt = PromptTemplate(input_variables=["blog_style", "input_text", "no_of_words"],
                            template=template)
    response = llm(prompt.format(blog_style=blog_style, input_text=input_text, no_of_words=no_of_words))
    st.write(response)
    return response


st.set_page_config(page_title= "Generate Blogs",
                   page_icon="🤖",
                   layout='centered',
                   initial_sidebar_state="collapsed")

st.header("Generate Blogs 🤖")

input_text = st.text_input("Enter the Blog Topic")

## creating two columns for additional two fields

col1, col2 = st.columns([5,5])

with col1:
    no_of_words = st.text_input("Enter the No. of words")
with col2:
    blog_style = st.selectbox("writing the blog for",
                              ('Researchers', 'Data Scientists', 'ML Engineers', 'Common People'),
                              index=0)
submit = st.button("Generate")

#final respponse
if submit:
    st.write(getLLamaResponse(input_text, no_of_words, blog_style))