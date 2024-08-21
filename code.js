figma.showUI(__html__, { width: 300, height: 300 });

figma.ui.onmessage = async (msg) => {
  if (msg.type === 'create-palette' && msg.data) {
    try {
      // Convert base64 image data to binary
      const base64Data = msg.data.split(',')[1];
      const byteCharacters = atob(base64Data);
      const byteNumbers = new Array(byteCharacters.length);

      for (let i = 0; i < byteCharacters.length; i++) {
        byteNumbers[i] = byteCharacters.charCodeAt(i);
      }

      const byteArray = new Uint8Array(byteNumbers);

      // Prepare the image for upload
      const formData = new FormData();
      formData.append('image', new Blob([byteArray], { type: 'image/png' }));

      // Send the image to the server
      const response = await fetch('http://localhost:5150/api/process-image', {
        method: 'POST',
        body: formData,
      });

      if (response.ok) {
        const result = await response.json();
        console.log('Model Result:', result.extended_color_palette);
        figma.ui.postMessage({ pluginMessage: { type: 'model-result', data: result.extended_color_palette } });
      } else {
        const errorText = await response.text();
        console.error('Server Error:', errorText);
        figma.ui.postMessage({ pluginMessage: { type: 'error', message: `Server Error: ${errorText}` } });
      }
    } catch (error) {
      console.error('Error:', error.message);
      figma.ui.postMessage({ pluginMessage: { type: 'error', message: error.message } });
    }
  }
};
