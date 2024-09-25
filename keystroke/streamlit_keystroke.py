import os
import streamlit as st
import streamlit.components.v1 as components

# Create a custom component to capture keystroke dynamics
def keystroke_capture():
    # Path to the JavaScript file we created earlier
    keystroke_js_path = os.path.join(os.path.dirname(__file__), "keystroke_capture.js")

    # Read the JavaScript file content
    with open(keystroke_js_path) as f:
        keystroke_js = f.read()

    # Create the custom Streamlit component
    component_value = components.html(
        f"""
        <script>
        {keystroke_js}
        </script>
        """, height=200
    )

    return component_value


# Streamlit app to test keystroke capture
st.title("Keystroke Dynamics Authentication")

st.write("Start typing in the box below:")

# Call the keystroke capture component
keystroke_data = keystroke_capture()

if keystroke_data:
    st.write("Keystroke Data Captured:", keystroke_data)
