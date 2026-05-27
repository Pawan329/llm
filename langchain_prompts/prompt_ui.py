from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import streamlit as st
from langchain_core.prompts import PromptTemplate

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

# template
template = PromptTemplate(
    template="""
please summarize the research paper titled "{paper_input}" with the following specifications:
Explanation style: {style_input}
Explanation length: {length_input}
1. Mathematical detail details:
- include relevant, mathematical equations if present in the paper.
- explain the mathematical concepts using simple, intuitive code, snippets, where applicable.
2. Analogies:
- use relatable, analogies to simplify complex ideas.
If certain information is not available in the paper, respond with: "insufficient information available" instead of guessing.
Ensure the summary is clear, accurate, and aligned with the provided style and length.
""",
input_variables=['paper_input', 'style_input', 'length_input']
)

#fill the placeholders in the template
prompt = template.invoke({
    'paper_input': paper_input,
    'style_input': style_input,
    'length_input': length_input
})


if st.button("Summarize"):
    response = model.invoke(prompt)
    st.write(response.content)