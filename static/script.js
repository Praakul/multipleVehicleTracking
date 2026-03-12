document.addEventListener('DOMContentLoaded', () => {
    // Elements
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');

    // Sections
    const sectionUpload = document.getElementById('upload-section');
    const sectionProcessing = document.getElementById('processing-section');
    const sectionResult = document.getElementById('result-section');

    // UI components
    const progressBar = document.getElementById('progress-bar');
    const statusText = document.getElementById('processing-status');
    const resultVideo = document.getElementById('result-video');
    const btnRestart = document.getElementById('btn-restart');
    const btnDownload = document.getElementById('btn-download');
    const btnPlay = document.getElementById('btn-play');

    // State management
    const switchSection = (newSection) => {
        document.querySelectorAll('.section').forEach(sec => sec.classList.remove('active'));
        newSection.classList.add('active');
    };

    // Drag and Drop Events
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => dropZone.classList.add('dragover'), false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => dropZone.classList.remove('dragover'), false);
    });

    dropZone.addEventListener('drop', (e) => {
        const files = e.dataTransfer.files;
        if (files.length) handleFile(files[0]);
    });

    fileInput.addEventListener('change', function () {
        if (this.files.length) handleFile(this.files[0]);
    });

    // Handle the selected file
    function handleFile(file) {
        if (!file.type.startsWith('video/')) {
            alert('Please select a valid video file.');
            return;
        }

        uploadAndProcess(file);
    }

    // Main API integration
    async function uploadAndProcess(file) {
        switchSection(sectionProcessing);

        // Fake progress bar animation (since HTTP POST doesn't natively stream progress for the response processing time easily)
        let progress = 0;
        const interval = setInterval(() => {
            progress += Math.random() * 5;
            if (progress > 90) progress = 90; // Stall at 90% until server responds
            progressBar.style.width = `${progress}%`;

            // Dynamic status text facts to keep users engaged
            if (progress > 20 && progress < 50) statusText.innerText = 'Detecting vehicles with YOLOv8...';
            if (progress > 50 && progress < 80) statusText.innerText = 'Correlating temporal boxes via SORT...';
            if (progress > 80) statusText.innerText = 'Applying Kalman Filter predictions...';
        }, 800);

        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch('/track_video/', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error(`Server returned ${response.status}`);
            }

            // Processing complete
            clearInterval(interval);
            progressBar.style.width = '100%';
            statusText.innerText = 'Processing Complete!';

            // Get the video blob
            const videoBlob = await response.blob();
            const videoUrl = URL.createObjectURL(videoBlob);

            // Wait a tiny bit for the 100% animation to finish smoothly
            setTimeout(() => {
                showResult(videoUrl, file.name);
            }, 500);

        } catch (error) {
            clearInterval(interval);
            alert(`Error processing video: ${error.message}`);
            switchSection(sectionUpload);
        }
    }

    function showResult(blobUrl, originalFileName) {
        // Set video source
        resultVideo.src = blobUrl;

        // Setup download button
        const safeName = originalFileName.replace(/\.[^/.]+$/, ""); // strip extension
        btnDownload.href = blobUrl;
        btnDownload.download = `tracked_${safeName}.webm`;
        
        // Setup play button
        if (btnPlay) {
            btnPlay.href = blobUrl;
        }

        switchSection(sectionResult);
    }

    // Restart flow
    btnRestart.addEventListener('click', () => {
        // Clear old video to free memory
        URL.revokeObjectURL(resultVideo.src);
        resultVideo.src = '';
        progressBar.style.width = '0%';
        statusText.innerText = 'Running YOLOv8 inference & SORT algorithm.';
        fileInput.value = ''; // reset file input
        switchSection(sectionUpload);
    });
});
