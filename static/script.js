document.getElementById('churnForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    // Dynamically extract all form data without hardcoding individual fields
    const formElement = document.getElementById('churnForm');
    const formData = new FormData(formElement);
    const payload = Object.fromEntries(formData.entries());

    // Typecast numerical values required by the backend
    payload.tenure = parseInt(payload.tenure);
    payload.MonthlyCharges = parseFloat(payload.MonthlyCharges);
    payload.SeniorCitizen = parseInt(payload.SeniorCitizen);
    
    // Auto-calculate TotalCharges based on user input
    payload.TotalCharges = payload.tenure * payload.MonthlyCharges;

    // Feature engineering fields expected by the updated model
    payload.Tenure_to_Monthly_Ratio = payload.tenure / (payload.MonthlyCharges + 1);
    payload.Has_Internet = payload.InternetService === "No" ? 0 : 1;

    try {
        const res = await fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        
        const data = await res.json();

        // Update UI dynamically
        document.getElementById('resultBox').innerHTML = `
            <div class="gauge" style="color: ${data.badge_color}">${data.churn_probability}%</div>
            <div class="badge" style="background: ${data.badge_color}22; color: ${data.badge_color}; border: 1px solid ${data.badge_color}">
                ${data.risk_level}
            </div>
            <div class="meta">
                <p>System Threshold: <strong>${data.operating_threshold}%</strong></p>
                <p style="margin-top: 8px;">${data.actionable_insight}</p>
            </div>
        `;
    } catch (error) {
        document.getElementById('resultBox').innerHTML = `<p style="color: #ef4444;">API Error: Backend offline.</p>`;
    }
});