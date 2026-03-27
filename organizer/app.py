import gradio as gr
from inference_custom import organizer_custom
from inference_groq import organizer_groq

def compare_models(user_input):
    # 1. Get results from both modules
    custom_results = organizer_custom(user_input)
    groq_results = organizer_groq(user_input)
    
    # 2. Return them to the UI
    return custom_results, groq_results

# Create the Interface
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 💸 Teeny Tiny Expense Organizer")
    gr.Markdown("Type your expenses below (e.g., 'Uber 200 and Swiggy 500') to see how the models categorize them.")
    
    with gr.Row():
        input_text = gr.Textbox(label="Enter Expenses", placeholder="50 for coffee, 100 for gas...")
    
    submit_btn = gr.Button("Organize!", variant="primary")
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 🧠 Custom DistilBERT (Your Model)")
            custom_output = gr.JSON(label="Custom Model Output")
            
        with gr.Column():
            gr.Markdown("### ☁️ Groq Llama-3 (The Giant)")
            groq_output = gr.JSON(label="Groq Output")

    submit_btn.click(
        fn=compare_models, 
        inputs=input_text, 
        outputs=[custom_output, groq_output]
    )

# Launch!
if __name__ == "__main__":
    demo.launch(share=True)