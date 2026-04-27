# app.py — updated for Gradio 6.0
import gradio as gr
from rag_pipeline import ask

def format_sources(metadatas):
    seen = []
    for meta in metadatas:
        subject = meta["subject"]
        if subject not in seen:
            seen.append(subject)
    return "Sources: " + ", ".join(seen)

def chat(question, history):
    if not question.strip():
        return "", history
    
    answer, chunks, metadatas = ask(question, verbose=False)
    sources = format_sources(metadatas)
    full_response = f"{answer}\n\n_{sources}_"
    
    # Gradio 6.0 format — dictionaries with role and content
    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": full_response})
    
    return "", history

with gr.Blocks(title="CineQuery") as demo:
    
    gr.Markdown("""
    # 🎬 CineQuery
    ### Your local cinema knowledge assistant
    Ask me about directors, films, acting styles, themes, or recommendations.
    *Running fully offline on your Mac — no internet required.*
    """)
    
    chatbot = gr.Chatbot(
        label="CineQuery",
        height=500,
        show_label=False,
    )
    
    with gr.Row():
        question_box = gr.Textbox(
            placeholder="e.g. What should I watch if I love Kubrick?",
            label="Your question",
            scale=4,
            lines=1,
        )
        submit_btn = gr.Button("Ask", variant="primary", scale=1)
    
    gr.Examples(
        examples=[
            "How does Scorsese use music in his films?",
            "What are the themes in Satyajit Ray's work?",
            "Recommend something if I love mind-bending narratives",
            "Tell me about Irrfan Khan's acting style",
            "What connects Kubrick and Villeneuve as directors?",
            "Tell me about Anurag Kashyap's filmmaking style",
            "How is Shah Rukh Khan different from other Bollywood actors?",
            "What should I watch if I liked The Godfather?",
        ],
        inputs=question_box,
    )
    
    clear_btn = gr.Button("Clear conversation", variant="secondary")
    
    question_box.submit(chat, inputs=[question_box, chatbot], outputs=[question_box, chatbot])
    submit_btn.click(chat, inputs=[question_box, chatbot], outputs=[question_box, chatbot])
    clear_btn.click(lambda: [], outputs=[chatbot])

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )