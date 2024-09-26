(function() {
    let startTime = null;
    let keyTimes = [];
  
    // Capture keypress and keyup events
    document.addEventListener('keypress', function(event) {
      let keyPressTime = new Date().getTime();
      if (!startTime) startTime = keyPressTime;
  
      let keyData = {
        key: event.key,
        keyPressTime: keyPressTime - startTime
      };
      keyTimes.push(keyData);
    });
  
    document.addEventListener('keyup', function(event) {
      let keyUpTime = new Date().getTime();
      let keyData = {
        key: event.key,
        keyUpTime: keyUpTime - startTime
      };
      keyTimes.push(keyData);
  
      // Send the captured data back to Streamlit
      Streamlit.setComponentValue(keyTimes);
    });
  })();
  