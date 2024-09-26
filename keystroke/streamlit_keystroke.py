import streamlit as st
from keystroke_capture import keystroke_capture

# Streamlit app to test keystroke capture
st.title("Keystroke Dynamics Authentication")

st.write("Start typing in the box below:")

# Call the keystroke capture component
keystroke_data = keystroke_capture()

# Ensure that we only process keystroke data after it is captured
if keystroke_data:
    try:
        # Display captured keystroke data
        st.write("Keystroke Data Captured:", keystroke_data)

        # Example: Analyze the time between key presses (Flight Time)
        key_press_times = [entry['keyPressTime'] for entry in keystroke_data if 'keyPressTime' in entry]
        
        if len(key_press_times) > 1:
            flight_times = [key_press_times[i+1] - key_press_times[i] for i in range(len(key_press_times)-1)]
            st.write("Flight Times (time between key presses):", flight_times)

        # Example: Analyze Dwell Time (time key is held down)
        dwell_times = []
        for i, entry in enumerate(keystroke_data):
            if 'keyPressTime' in entry and 'keyUpTime' in entry:
                dwell_times.append(entry['keyUpTime'] - entry['keyPressTime'])
        st.write("Dwell Times (time key is held down):", dwell_times)

    except Exception as e:
        st.error(f"Error processing keystroke data: {e}")
