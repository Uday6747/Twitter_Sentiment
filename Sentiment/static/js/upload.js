




document.querySelector('form').addEventListener('submit', async function(event) {
    event.preventDefault(); // Prevent default form submission

    let formData = new FormData(this); // Create FormData object from the form

    try {
        let response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });

        if (response.ok) {
            // Upload successful
            console.log('File uploaded successfully!');
            // Perform any additional actions if needed
        } else {
            // Upload failed
            console.error('File upload failed.');
        }
    } catch (error) {
        console.error('An error occurred during file upload:', error);
    }
});

document.getElementById('file-upload').addEventListener('change', function() {
    var fileName = this.files[0].name;
    document.getElementById('file-name').textContent = fileName;

    // Enable/disable the upload button based on whether a file is selected
    let uploadBtn = document.querySelector(".upload__button");
    if (fileName !== "No file chosen...") {
        let sleep = (time) => new Promise(resolve => setTimeout(resolve, time));
        let upload = document.querySelector(".upload");
        let uploadBtn = document.querySelector(".upload__button");
        uploadBtn.addEventListener("click", async () => {
        upload.classList.add("uploading");
        await sleep(3000);
        upload.classList.add("uploaded");
        await sleep(2000);
        upload.classList.remove("uploading");
        upload.classList.add("uploaded-after");
        await sleep(1000);
        upload.className = "upload";
        });
    } else {
        uploadBtn.disabled = true;
    }
});

document.querySelector('.result-button').addEventListener('click', function() {
    let fileName = document.getElementById('file-name').textContent;
    if (fileName === "No file chosen...") {
        alert("Please upload a file before proceeding.");
    } else {
        const resultButton = document.getElementById('resultButton');

        // Add click event listener to the Result button
        resultButton.addEventListener('click', function() {
            // Redirect to result.html
            window.location.href = '/file_result';
        });
    }
});
