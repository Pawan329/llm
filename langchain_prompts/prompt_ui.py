from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import streamlit as st
from langchain_core.prompts import PromptTemplate, load_prompt

load_dotenv()
model = ChatOpenAI()

st.header("Reasearch Tool")
paper_input = st.selectbox("select research paper name", 
                           ["Select...",
                            "Attention is all you need",
                            "BERT: retraining of deep by directional Transformers",
                            "GPT-3: language models are few learners",
                            "diffusion models beat GANs on image synthesis" ])

style_input = st.selectbox("select explanation style", 
                           ["beginner, friendly", "technical","code oriented","mathematical"])

length_input = st.selectbox("select explanation length",
                           ["short (1-2 sentences)", "medium (1-2 paragraphs)", "long (detailed explanation)"])


#load template form json file
template = load_prompt('../template.json')

#fill the placeholders in the template
prompt = template.invoke({
    'paper_input': paper_input,
    'style_input': style_input,
    'length_input': length_input
})


if st.button("Summarize"):
    response = model.invoke(prompt)
    st.write(response.content)