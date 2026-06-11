document.addEventListener('DOMContentLoaded', () => {
    // Sliders list
    const sliders = [
        { id: 'edad', valId: 'edad-val', suffix: '' },
        { id: 'años_en_empresa', valId: 'años_en_empresa-val', suffix: '' },
        { id: 'salario_mensual', valId: 'salario_mensual-val', prefix: '$', suffix: '' },
        { id: 'horas_extra_semana', valId: 'horas_extra_semana-val', suffix: '' },
        { id: 'num_proyectos_año', valId: 'num_proyectos_año-val', suffix: '' },
        { id: 'distancia_casa_trabajo_km', valId: 'distancia_casa_trabajo_km-val', suffix: ' km' },
        { id: 'ultima_evaluacion_desempeño', valId: 'ultima_evaluacion_desempeño-val', suffix: '' },
        { id: 'capacitaciones_recibidas', valId: 'capacitaciones_recibidas-val', suffix: '' }
    ];

    // Initialize and setup input listeners for sliders
    sliders.forEach(slider => {
        const inputEl = document.getElementById(slider.id);
        const valEl = document.getElementById(slider.valId);
        
        if (inputEl && valEl) {
            const updateValue = () => {
                let valueText = inputEl.value;
                if (slider.prefix) valueText = slider.prefix + valueText;
                if (slider.suffix) valueText = valueText + slider.suffix;
                valEl.textContent = valueText;
            };
            
            inputEl.addEventListener('input', updateValue);
            updateValue(); // Initial state
        }
    });

    const form = document.getElementById('prediction-form');
    const placeholderResult = document.getElementById('placeholder-result');
    const realResult = document.getElementById('real-result');
    const riskGauge = document.getElementById('risk-gauge');
    const riskPercentage = document.getElementById('risk-percentage');
    const riskBadge = document.getElementById('risk-badge');
    const predictionSummary = document.getElementById('prediction-summary');
    const recommendationsList = document.getElementById('recommendations-list');

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        // Gather values from form
        const formData = new FormData(form);
        const payload = {};
        
        formData.forEach((value, key) => {
            payload[key] = value;
        });

        // Add sliders that might not have standard name behavior if any, 
        // but FormData captures input elements with name attribute correctly.
        // Let's verify we have name attributes on all inputs.
        // We do: edad, años_en_empresa, salario_mensual, horas_extra_semana, num_proyectos_año, 
        // distancia_casa_trabajo_km, ultima_evaluacion_desempeño, capacitaciones_recibidas,
        // satisfaccion_laboral, tiene_ascenso_ultimos_2_años.
        
        // Show loading state or block UI
        const btn = form.querySelector('.btn-predict');
        const originalBtnText = btn.innerHTML;
        btn.disabled = true;
        btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Calculando...';

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(payload)
            });
            
            const data = await response.json();
            
            if (response.ok) {
                // Hide placeholder and show result
                placeholderResult.classList.add('hidden');
                realResult.classList.remove('hidden');
                
                // Get probability and risk
                const prob = data.probabilidad_renuncia;
                const pct = Math.round(prob * 100);
                const riesgo = data.riesgo; // 'Bajo', 'Medio', 'Alto'
                
                // Set percentage in gauge
                riskGauge.style.setProperty('--risk-percent', pct);
                riskPercentage.textContent = pct + '%';
                
                // Color mapping based on risk level
                let riskColor = '#10b981'; // Green
                let badgeClass = 'low';
                
                if (riesgo === 'Alto') {
                    riskColor = '#ef4444'; // Red
                    badgeClass = 'high';
                } else if (riesgo === 'Medio') {
                    riskColor = '#f59e0b'; // Orange
                    badgeClass = 'medium';
                }
                
                riskGauge.style.setProperty('--risk-color', riskColor);
                
                // Set badge
                riskBadge.textContent = riesgo;
                riskBadge.className = 'badge ' + badgeClass;
                
                // Set prediction summary text
                if (data.renuncia === 1) {
                    predictionSummary.innerHTML = `El sistema predice que el empleado <strong>RENUNCIARÁ</strong> en los próximos 6 meses (Probabilidad: <strong>${pct}%</strong>).`;
                } else {
                    predictionSummary.innerHTML = `El sistema predice que el empleado <strong>SE QUEDARÁ</strong> en la empresa (Probabilidad de renuncia: <strong>${pct}%</strong>).`;
                }
                
                // Populate recommendations
                recommendationsList.innerHTML = '';
                data.recomendaciones.forEach(rec => {
                    const li = document.createElement('li');
                    li.textContent = rec;
                    li.style.setProperty('--risk-color', riskColor);
                    recommendationsList.appendChild(li);
                });
                
                // Scroll to result on small screens
                if (window.innerWidth <= 900) {
                    realResult.scrollIntoView({ behavior: 'smooth' });
                }
            } else {
                alert('Error en la predicción: ' + (data.error || 'Intente de nuevo.'));
            }
        } catch (error) {
            console.error('Error:', error);
            alert('Error de red al conectar con el servidor.');
        } finally {
            btn.disabled = false;
            btn.innerHTML = originalBtnText;
        }
    });
});
