// MindSpace — Micro Interactions & UI Enhancements

document.addEventListener('DOMContentLoaded', () => {

  // ── Staggered card entrance on assessment page
  const qCards = document.querySelectorAll('.q-card');
  if (qCards.length) {
    qCards.forEach((card, i) => {
      card.style.opacity = '0';
      card.style.transform = 'translateY(10px)';
      card.style.transition = `opacity 0.3s ease ${i * 0.02}s, transform 0.3s ease ${i * 0.02}s, border-color 0.2s, box-shadow 0.2s`;
      requestAnimationFrame(() => {
        setTimeout(() => {
          card.style.opacity = '1';
          card.style.transform = 'translateY(0)';
        }, 50 + i * 20);
      });
    });
  }

  // ── Animate probability bars on result page
  const probFills = document.querySelectorAll('.prob-fill');
  if (probFills.length) {
    probFills.forEach(bar => {
      const target = bar.style.width;
      bar.style.width = '0%';
      setTimeout(() => { bar.style.width = target; }, 300);
    });
  }

  // ── Animate rec cards on result page
  const recCards = document.querySelectorAll('.rec-card');
  recCards.forEach((card, i) => {
    card.style.opacity = '0';
    card.style.transform = 'translateY(16px)';
    card.style.transition = `opacity 0.4s ease ${0.5 + i * 0.07}s, transform 0.4s ease ${0.5 + i * 0.07}s, box-shadow 0.2s, border-color 0.2s`;
    setTimeout(() => {
      card.style.opacity = '1';
      card.style.transform = 'translateY(0)';
    }, 100);
  });

  // ── Chart cards entrance
  const chartCards = document.querySelectorAll('.chart-card');
  chartCards.forEach((card, i) => {
    card.style.opacity = '0';
    card.style.transform = 'translateY(12px)';
    card.style.transition = `opacity 0.5s ease ${0.2 + i * 0.1}s, transform 0.5s ease ${0.2 + i * 0.1}s, box-shadow 0.2s, transform 0.2s`;
    setTimeout(() => {
      card.style.opacity = '1';
      card.style.transform = 'translateY(0)';
    }, 100);
  });

  // ── Score ring animation entrance
  const ring = document.querySelector('.ring-fill');
  if (ring) {
    const finalDash = ring.getAttribute('stroke-dasharray');
    ring.setAttribute('stroke-dasharray', '0 502.65');
    setTimeout(() => { ring.style.transition = 'stroke-dasharray 1.4s cubic-bezier(0.34,1.56,0.64,1)'; ring.setAttribute('stroke-dasharray', finalDash); }, 400);
  }

  // ── Input focus label animation
  document.querySelectorAll('.form-input').forEach(inp => {
    inp.addEventListener('focus', () => inp.parentElement.classList.add('focused'));
    inp.addEventListener('blur',  () => inp.parentElement.classList.remove('focused'));
  });

  // ── Assessment form validation highlight
  const assessForm = document.getElementById('assessForm');
  if (assessForm) {
    assessForm.addEventListener('submit', function(e) {
      let allAnswered = true;
      for (let i = 1; i <= 20; i++) {
        const checked = document.querySelector(`input[name="q${i}"]:checked`);
        const card = document.querySelector(`.q-card[data-q="${i}"]`);
        if (!checked) {
          allAnswered = false;
          if (card) {
            card.style.animation = 'shake 0.4s ease';
            card.style.borderColor = '#EF4444';
            setTimeout(() => { card.style.animation = ''; card.style.borderColor = ''; }, 600);
          }
        }
      }
      if (!allAnswered) {
        e.preventDefault();
        document.querySelector('.analyze-note').textContent = '⚠️ Please answer all 20 questions before analyzing.';
        document.querySelector('.analyze-note').style.color = '#EF4444';
      }
    });
  }
});

// Add shake animation dynamically
const shakeStyle = document.createElement('style');
shakeStyle.textContent = `
@keyframes shake {
  0%,100% { transform: translateX(0); }
  20%      { transform: translateX(-5px); }
  40%      { transform: translateX(5px); }
  60%      { transform: translateX(-4px); }
  80%      { transform: translateX(4px); }
}`;
document.head.appendChild(shakeStyle);
