import streamlit.components.v1 as components

# Declare the custom component, which will use the JavaScript for keystroke capture
keystroke_component = components.declare_component(
    "keystroke_capture",
    path="./"  # Use the local JavaScript component here
)

# Function to run the custom component and capture the keystroke dynamics
def keystroke_capture():
    # Call the JavaScript component with empty data initially
    component_value = keystroke_component()

    # Return the value (the keystroke data)
    return component_value
