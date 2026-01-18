import gradio as gr

def launch_ui(process_fn, chat_fn):
    """
    Finance Insight Lite - UI Module
    This function defines the layout and connects to the backend logic.
    """
    
    # Define the theme separately to avoid constructor warnings in new Gradio versions
    custom_theme = gr.themes.Soft(
        primary_hue="emerald",
        neutral_hue="slate",
    )

    with gr.Blocks(title="Finance Insight Lite") as demo:
        # Professional App Header
        gr.HTML(
            """
            <div style='text-align: center; padding: 15px;'>
                <h1 style='color: #10b981; margin: 0;'>📊 Finance Insight Lite</h1>
                <p style='color: #64748b; font-size: 1.1em;'>Corporate Financial Intelligence Engine</p>
            </div>
            """
        )
        
        with gr.Row():
            # Control Panel (Left)
            with gr.Column(scale=1, variant="panel"):
                gr.Markdown("### 🛠️ **Command Center**")
                file_input = gr.File(label="Upload Financial Report (PDF)", file_types=[".pdf"])
                process_btn = gr.Button("INITIALIZE ANALYSIS", variant="primary")
                status_output = gr.Textbox(label="System Status", interactive=False)
                
                gr.Markdown("---")
                gr.Markdown("### 💡 **Quick Queries**")
                
                # Chat input placed here for better UX in Example selection
                msg_input = gr.Textbox(
                    label="Input Query", 
                    placeholder="Ask about financial metrics...",
                    show_label=True
                )
                
                gr.Examples(
                    examples=[
                        "Summarize the key financial highlights.",
                        "What is the total equity for 2023?",
                        "Extract the dividend policy details.",
                        "Analyze the net income trends."
                    ],
                    inputs=msg_input
                )

            # Chat Interface (Right)
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(label="Bayan AI Terminal", height=600)
                submit_btn = gr.Button("SEND QUERY", variant="primary")

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
        server_name="0.0.0.0"
    )