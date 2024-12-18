function updateFileName(input) {
    const fileName = input.files[0].name;
    document.getElementById('fileName').textContent = fileName;
}

// Hide the upload form if the original image is displayed
function hideUploadForm(originalImage