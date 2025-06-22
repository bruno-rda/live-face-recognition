import gradio as gr
from services.face_service import FaceService
from config import Config

# State Initialization
def init_state():
    '''
    Initializes the application state including the face service.
    This function will be called once when the Gradio app starts.
    '''
    print('Initializing FaceService...')
    service = FaceService()
    print('FaceService initialization complete.')
    
    state = {
        'face_service': service, # Store the single instance of FaceService
        'last_reg_status': '',
        'last_reg_success': False,
        'last_manage_status': '',
        'last_manage_success': False,
    }
    print(f'Loaded {service.get_registered_count()} embeddings.')
    return state

# Gradio Specific Functions (minimal logic, delegates to service)

def process_frame_predict_gradio(state, frame):
    """Processes frame for prediction mode using the stored FaceService instance."""
    return state['face_service'].process_frame_for_prediction(frame)

def process_frame_register_gradio(state, frame):
    """Processes frame for registration preview using the stored FaceService instance."""
    return state['face_service'].process_frame_for_registration_preview(frame)

def register_face_gradio(state, name, frame_snapshot):
    """Attempts to register a face using the stored FaceService instance."""
    status_message, success = state['face_service'].register_new_face(name, frame_snapshot)
    state['last_reg_status'] = status_message
    state['last_reg_success'] = success
    return state

def show_register_feedback(state):
    """Returns the last registration status message for Gradio toast."""
    status = state['last_reg_status']
    if not status:
        return None
    elif status.startswith('Success'):
        return gr.Success(status)
    elif status.startswith('Error'):
        raise gr.Error(status)
    else:
        return gr.Info(status)

def get_registered_names_gradio(state):
    """Retrieves all registered names using the stored FaceService instance."""
    return state['face_service'].get_all_registered_names()

def update_manage_dropdown_gradio(state):
    """Updates the dropdown choices for registered names."""
    names = get_registered_names_gradio(state) # Use the helper that accesses state
    return gr.update(choices=names, value=None)

def rename_face_entry_gradio(state, old_name, new_name, password):
    """Renames a face entry using the stored FaceService instance."""
    status_message, success = state['face_service'].rename_existing_face(old_name, new_name, password)
    state['last_manage_status'] = status_message
    state['last_manage_success'] = success
    return state

def delete_face_entry_gradio(state, name_to_delete, password):
    """Deletes a face entry using the stored FaceService instance."""
    status_message, success = state['face_service'].delete_existing_face(name_to_delete, password)
    state['last_manage_status'] = status_message
    state['last_manage_success'] = success
    return state

def show_manage_feedback(state):
    """Returns the last management status message for Gradio toast."""
    status = state['last_manage_status']
    if not status:
        return None
    elif status.startswith('Success'):
        return gr.Success(status)
    elif status.startswith('Error'):
        raise gr.Error(status)
    else:
        return gr.Info(status)


# Build Gradio App

with gr.Blocks(theme=gr.themes.Monochrome()) as demo:
    # Initialize state once when the app starts
    app_state = gr.State(value=init_state())

    gr.Markdown('# Face Recognition App')
    gr.Markdown('Use the tabs below to switch between modes.')

    with gr.Tabs():
        # Prediction Tab
        with gr.TabItem('Predict Faces'):
            with gr.Row():
                with gr.Column():
                    predict_camera_input = gr.Image(
                        label='Camera Feed', streaming=True, type='numpy', height=480, width=640
                    )
                with gr.Column():
                    predict_processed_output = gr.Image(
                        label='Recognized Faces', type='numpy', height=480, width=640
                    )
            predict_camera_input.stream(
                fn=process_frame_predict_gradio,
                inputs=[app_state, predict_camera_input],
                outputs=predict_processed_output,
                stream_every=Config.STREAM_INTERVAL
            )

        # Registration Tab 
        with gr.TabItem('Register New Face'):
            with gr.Row():
                with gr.Column():
                    register_camera_input = gr.Image(
                        label='Camera Feed (for Registration)', streaming=True, type='numpy', height=480, width=640
                    )
                    register_status_display = gr.Markdown(value='Status: Not registered yet.')
                    name_textbox = gr.Textbox(
                        label='Name', placeholder='Enter name to register'
                    )
                    register_button = gr.Button('Register Face from Preview', variant='primary')
                with gr.Column():
                    register_processed_output = gr.Image(
                        label='Face Detection Preview', type='numpy', height=480, width=640, interactive=False
                    )

            register_camera_input.stream(
                fn=process_frame_register_gradio,
                inputs=[app_state, register_camera_input],
                outputs=register_processed_output,
                stream_every=Config.STREAM_INTERVAL
            )

            register_button.click(
                fn=register_face_gradio,
                inputs=[app_state, name_textbox, register_processed_output],
                outputs=[app_state]
            ).then(
                fn=show_register_feedback,
                inputs=app_state,
                outputs=None
            ).then(
                fn=lambda state: gr.update(value=f"**Status:** {state['last_reg_status']}"),
                inputs=app_state,
                outputs=register_status_display
            ).then(
                fn=lambda state: gr.update(value='') if state['last_reg_success'] else gr.update(),
                inputs=app_state,
                outputs=name_textbox
            )

        # Manage Faces Tab 
        with gr.TabItem('Manage Faces') as manage_tab:
            with gr.Row():
                 gr.Markdown('Select a face to rename or delete.')
            with gr.Row():
                manage_face_selector = gr.Dropdown(
                    label='Registered Faces',
                    # Pass app_state to initial choices call
                    choices=get_registered_names_gradio(app_state.value), # Initial choices
                    interactive=True
                )
            with gr.Row():
                rename_textbox = gr.Textbox(label='New Name (for renaming)', placeholder='Enter new name here')
            with gr.Row():
                password_textbox = gr.Textbox(label='Password', placeholder='Enter password here', type='password')
            with gr.Row():
                rename_button = gr.Button('Rename Selected Face', variant='primary')
                delete_button = gr.Button('Delete Selected Face')

            rename_button.click(
                fn=rename_face_entry_gradio,
                inputs=[app_state, manage_face_selector, rename_textbox, password_textbox],
                outputs=[app_state]
            ).then(
                # Pass app_state to update dropdown
                fn=update_manage_dropdown_gradio,
                inputs=app_state,
                outputs=manage_face_selector,
                # Conditionally update if success is needed, though updating always is safer for UI consistency
                # fn=lambda state: update_manage_dropdown_gradio(state) if state['last_manage_success'] else gr.update(),
            ).then(
                fn=lambda state: gr.update(value='') if state['last_manage_success'] else gr.update(),
                inputs=app_state,
                outputs=rename_textbox
            ).then(
                fn=show_manage_feedback,
                inputs=app_state,
                outputs=None
            )

            delete_button.click(
                fn=delete_face_entry_gradio,
                inputs=[app_state, manage_face_selector, password_textbox],
                outputs=[app_state]
            ).then(
                # Pass app_state to update dropdown
                fn=update_manage_dropdown_gradio,
                inputs=app_state,
                outputs=manage_face_selector,
                # Conditionally update if success is needed
                # fn=lambda state: update_manage_dropdown_gradio(state) if state['last_manage_success'] else gr.update(),
            ).then(
                 fn=lambda state: gr.update(value='') if state['last_manage_success'] else gr.update(),
                 inputs=app_state,
                 outputs=rename_textbox
            ).then(
                fn=show_manage_feedback,
                inputs=app_state,
                outputs=None
            )

            # Refresh dropdown when the tab is selected
            manage_tab.select(
                 fn=update_manage_dropdown_gradio,
                 inputs=app_state, # Pass app_state here
                 outputs=manage_face_selector
            )

# Launch the app
if __name__ == '__main__':
    print('Launching Gradio app...')
    demo.launch()