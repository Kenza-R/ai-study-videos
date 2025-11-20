// Upload Page Interactivity

document.addEventListener('DOMContentLoaded', function() {
    // Elements
    const fileInputs = document.querySelectorAll('input[type="file"]');
    const fileUploadArea = document.querySelector('.file-upload-area');
    const fileNameDisplay = document.querySelector('.file-name');
    const form = document.querySelector('form');
    const submitBtn = document.querySelector('.btn-primary');
    const textInputs = document.querySelectorAll('input[type="text"]');

    // Drag and drop functionality
    if (fileUploadArea) {
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            fileUploadArea.addEventListener(eventName, preventDefaults, false);
        });

        function preventDefaults(e) {
            e.preventDefault();
            e.stopPropagation();
        }

        ['dragenter', 'dragover'].forEach(eventName => {
            fileUploadArea.addEventListener(eventName, highlight, false);
        });

        ['dragleave', 'drop'].forEach(eventName => {
            fileUploadArea.addEventListener(eventName, unhighlight, false);
        });

        function highlight(e) {
            fileUploadArea.classList.add('drag-over');
        }

        function unhighlight(e) {
            fileUploadArea.classList.remove('drag-over');
        }

        fileUploadArea.addEventListener('drop', handleDrop, false);

        function handleDrop(e) {
            const dt = e.dataTransfer;
            const files = dt.files;
            handleFiles(files);
        }

        fileUploadArea.addEventListener('click', () => {
            const fileInput = document.querySelector('input[type="file"]');
            if (fileInput) {
                fileInput.click();
            }
        });
    }

    // Handle file selection
    fileInputs.forEach(input => {
        input.addEventListener('change', function() {
            handleFiles(this.files);
        });
    });

    function handleFiles(files) {
        if (files.length > 0) {
            const file = files[0];
            const fileName = file.name;
            const fileSize = (file.size / 1024 / 1024).toFixed(2); // Convert to MB

            if (fileNameDisplay) {
                fileNameDisplay.innerHTML = `
                    <span style="color: #10b981;">✓</span>
                    <strong>${fileName}</strong> (${fileSize} MB)
                `;
            }

            // Add visual feedback
            if (fileUploadArea) {
                fileUploadArea.style.borderColor = '#10b981';
                fileUploadArea.style.background = 'rgba(16, 185, 129, 0.05)';
            }
        }
    }

    // Form validation
    if (form) {
        form.addEventListener('submit', function(e) {
            const textInputValue = Array.from(textInputs).some(input => input.value.trim());
            const fileInputValue = Array.from(fileInputs).some(input => input.files.length > 0);

            if (!textInputValue && !fileInputValue) {
                e.preventDefault();
                showAlert('Please provide either a PubMed ID or upload a PDF file.', 'error');
            }
        });
    }

    // Input focus effects
    textInputs.forEach(input => {
        input.addEventListener('focus', function() {
            this.parentElement.classList.add('focused');
        });

        input.addEventListener('blur', function() {
            this.parentElement.classList.remove('focused');
        });
    });

    // Button loading state
    if (submitBtn && form) {
        form.addEventListener('submit', function(e) {
            const textInputValue = Array.from(textInputs).some(input => input.value.trim());
            const fileInputValue = Array.from(fileInputs).some(input => input.files.length > 0);

            if (textInputValue || fileInputValue) {
                submitBtn.disabled = true;
                submitBtn.classList.add('loading');
                const originalText = submitBtn.textContent;
                submitBtn.textContent = 'Processing...';

                // Reset after 3 seconds if form doesn't submit
                setTimeout(() => {
                    submitBtn.disabled = false;
                    submitBtn.classList.remove('loading');
                    submitBtn.textContent = originalText;
                }, 3000);
            }
        });
    }

    // Show alert messages
    function showAlert(message, type) {
        const alertDiv = document.createElement('div');
        alertDiv.className = `alert alert-${type}`;
        alertDiv.textContent = message;

        const formGroup = document.querySelector('.form-group');
        if (formGroup) {
            formGroup.parentElement.insertBefore(alertDiv, formGroup);

            // Auto remove after 5 seconds
            setTimeout(() => {
                alertDiv.style.animation = 'slideUp 0.3s ease-out forwards';
                setTimeout(() => alertDiv.remove(), 300);
            }, 5000);
        }
    }

    // Smooth scroll to errors
    const errorMessages = document.querySelectorAll('.errorlist');
    if (errorMessages.length > 0) {
        errorMessages.forEach(error => {
            error.scrollIntoView({ behavior: 'smooth', block: 'center' });
        });
    }

    // Add smooth animations to form groups
    const formGroups = document.querySelectorAll('.form-group');
    formGroups.forEach((group, index) => {
        group.style.animation = `slideUp 0.4s ease-out ${index * 0.1}s backwards`;
    });
});
