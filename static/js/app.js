// Client-side utilities for attribution analysis
document.addEventListener('DOMContentLoaded', function() {
    // File input validation
    const uploadForm = document.getElementById('uploadForm');
    if (uploadForm) {
        uploadForm.addEventListener('submit', function(e) {
            const benchmark = document.getElementById('benchmark_file');
            const portfolio = document.getElementById('portfolio_file');
            const useNcreif = document.getElementById('useNcreif');
            const usingCache = useNcreif && useNcreif.value === '1';
            if ((!usingCache && !benchmark.value) || !portfolio.value) {
                e.preventDefault();
                alert('Please select the required files.');
                return false;
            }
            // Show loading indicator
            const btn = uploadForm.querySelector('button[type="submit"]');
            btn.disabled = true;
            btn.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>Processing...';
        });
    }
});
