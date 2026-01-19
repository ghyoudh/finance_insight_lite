import gradio as gr
import os
import base64

def get_base64_img(img_path):
    try:
        with open(img_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        print(f"Error loading image: {e}")
        return ""


def launch_ui(process_fn, chat_fn):
    #load and encode the logo image
    img_data = get_base64_img("images/logo.png")
    
    custom_theme = gr.themes.Soft(primary_hue="emerald", neutral_hue="slate")
    
    with gr.Blocks(title="Finance Insight Lite") as demo:
        # --- Header Section ---
        gr.HTML(
            f"""
    <div style='text-align: left; padding: 20px 0; display: flex; align-items: center; gap: 25px;'>
        <div style='width: 130px; 
                    height: 130px; 
                    border-radius: 50%; 
                    overflow: hidden; 
                    border: 4px solid #FFD700; 
                    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
                    display: flex; 
                    align-items: center; 
                    justify-content: center;'>
            
            <img src='data:image/png;base64,{img_data}' 
                 style='width: 100%; 
                        height: 100%; 
                        object-fit: cover; 
                        transform: scale(1.4); 
                        transform-origin: center;'>
        </div>
        
        <div>
            <h1 style='color: #10b981; margin: 0; font-size: 3em; font-weight: 800; font-family: sans-serif;'>
                Finance Insight Lite
            </h1>
            <p style='color: #64748b; font-size: 1.4em; margin: 5px 0 0 0; font-family: sans-serif;'>
                Corporate Financial Intelligence Engine
            </p>
        </div>
    </div>
    """
)
        
        
        with gr.Row():
            # Control Panel (Left)
            with gr.Column(scale=1, variant="panel"):
                gr.Markdown("### 🛠️ **Command Center**")
                file_input = gr.File(label="Upload Financial Report (PDF)", file_types=[".pdf"])
                process_btn = gr.Button("INITIALIZE ANALYSIS", variant="primary")
                status_output = gr.Markdown("", label="Status")

                

                gr.Markdown("---")
                gr.Markdown("""
                **Agent Instructions:**
                1. Upload the annual report.
                2. Wait for system initialization.
                3. Ask specific financial questions in the chat.
                """)
                
               

           # Right Panel: Chat Interface (Cleaner)
            with gr.Column(scale=2):
                # Chatbot component now includes the input bar inside it in newer Gradio versions
                # Or we place it immediately below for better control
                chatbot = gr.Chatbot(
                    label="AI Terminal", 
                    height=550, 
                )
                
                with gr.Row():
                    msg_input = gr.Textbox(
                        label=None,
                        placeholder="Type your financial query here...",
                        show_label=False,
                        scale=4
                    )
                    submit_btn = gr.Button("Send", variant="primary", scale=1)

                # Examples moved under the chat input for a cleaner look
                gr.Examples(
                    examples=[
                        "Summarize the key financial highlights.",
                        "What is the total equity for 2023?",
                        "Extract the dividend policy details.",
                        "Analyze the net income trends."
                    ],
                    inputs=msg_input,
                    label="Suggested Queries"
                )

        # --- Event Logic ---
        def handle_upload(file):
            if file is None: return "❌ No file selected."
            try:
                # Backend process_fn passed from app.py
                return process_fn(file.name)
            except Exception as e:
                return f"⚠️ Error: {str(e)}"

        def handle_chat(message, history):
            if not message.strip(): return "", history
            
            # Backend chat_fn passed from app.py
            response = chat_fn(message)
            bot_msg = response.get("answer") or response.get("result") or "No data retrieved."
            
            history.append((message, bot_msg))
            return "", history

        # --- Event Listeners ---
        process_btn.click(handle_upload, inputs=[file_input], outputs=[status_output])
        submit_btn.click(handle_chat, inputs=[msg_input, chatbot], outputs=[msg_input, chatbot])
        msg_input.submit(handle_chat, inputs=[msg_input, chatbot], outputs=[msg_input, chatbot])

    # Important: Moved 'theme' to launch() to resolve the UserWarning
    # Also added share=True to generate the public link for your team
    demo.launch(
        theme=custom_theme,
        share=True,
        server_name="0.0.0.0",
        inline=False,
        allowed_paths=[os.path.abspath("images")]
    )


if __name__ == "__main__":
    # For standalone testing purposes
    def dummy_process(file_path):
        return "✅ File processed successfully."

    def dummy_chat(message):
        return {"answer": f"Echo: {message}"}

    launch_ui(dummy_process, dummy_chat)