(function () {
  const uploadZone = document.getElementById('uploadZone');
  const fileInput = document.getElementById('fileInput');
  const btnPredictImage = document.getElementById('btnPredictImage');
  const btnPredictText = document.getElementById('btnPredictText');
  const subjectInput = document.getElementById('subjectInput');
  const bodyInput = document.getElementById('bodyInput');
  const resultCard = document.getElementById('resultCard');
  const resultLabel = document.getElementById('resultLabel');
  const resultConfidence = document.getElementById('resultConfidence');
  const resultMessage = document.getElementById('resultMessage');
  const resultPreview = document.getElementById('resultPreview');
  const resultProbChip = document.getElementById('resultProbChip');
  const resultScoreChip = document.getElementById('resultScoreChip');
  const imageError = document.getElementById('imageError');
  const textError = document.getElementById('textError');
  const uploadPreview = document.getElementById('uploadPreview');
  const uploadPreviewWrapper = uploadPreview && uploadPreview.parentElement;
  const progressCard = document.getElementById('progressCard');
  const progressLabelMain = document.getElementById('progressLabelMain');
  const progressLabelSub = document.getElementById('progressLabelSub');
  const progressBarFill = document.getElementById('progressBarFill');
  const progressPercent = document.getElementById('progressPercent');

  let selectedFile = null;
  let previewUrl = null;
  let progressTimer = null;
  let currentProgress = 0;

  function setProgress(value) {
    if (!progressBarFill || !progressPercent) return;
    const clamped = Math.max(0, Math.min(100, value));
    currentProgress = clamped;
    const scale = clamped / 100;
    progressBarFill.style.transform = 'scaleX(' + scale + ')';
    progressPercent.textContent = clamped.toFixed(0) + '%';
  }

  function startProgress(kind) {
    if (!progressCard || !progressBarFill || !progressPercent) return;
    if (progressTimer) {
      clearInterval(progressTimer);
      progressTimer = null;
    }
    currentProgress = 0;
    setProgress(0);
    if (progressLabelMain) {
      progressLabelMain.textContent =
        kind === 'image' ? 'Analyzing screenshot…' : 'Analyzing email text…';
    }
    if (progressLabelSub) {
      progressLabelSub.textContent = 'Running OCR (if needed) and phishing classifier.';
    }
    progressCard.classList.add('visible');

    // Increment smoothly up to 88%; final jump to 100% when response returns.
    progressTimer = setInterval(function () {
      if (currentProgress >= 88) return;
      const step = 3 + Math.random() * 4;
      setProgress(currentProgress + step);
    }, 220);
  }

  function finishProgress() {
    if (!progressCard) return;
    if (progressTimer) {
      clearInterval(progressTimer);
      progressTimer = null;
    }
    setProgress(100);
    setTimeout(function () {
      progressCard.classList.remove('visible');
      setProgress(0);
    }, 600);
  }

  function updatePreview(file) {
    if (!uploadPreview || !uploadPreviewWrapper) return;
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
      previewUrl = null;
    }
    if (!file || !file.type || !file.type.startsWith('image/')) {
      uploadPreviewWrapper.classList.remove('visible');
      uploadPreview.src = '';
      if (uploadZone) {
        uploadZone.classList.remove('has-preview');
      }
      return;
    }
    previewUrl = URL.createObjectURL(file);
    uploadPreview.src = previewUrl;
    uploadPreviewWrapper.classList.add('visible');
    if (uploadZone) {
      uploadZone.classList.add('has-preview');
    }
  }

  uploadZone.addEventListener('click', function () { fileInput.click(); });
  uploadZone.addEventListener('dragover', function (e) { e.preventDefault(); uploadZone.classList.add('dragover'); });
  uploadZone.addEventListener('dragleave', function () { uploadZone.classList.remove('dragover'); });
  uploadZone.addEventListener('drop', function (e) {
    e.preventDefault();
    uploadZone.classList.remove('dragover');
    const f = e.dataTransfer && e.dataTransfer.files[0];
    if (f && f.type && f.type.startsWith('image/')) {
      selectedFile = f;
      btnPredictImage.disabled = false;
      imageError.textContent = '';
      updatePreview(f);
    }
  });
  fileInput.addEventListener('change', function () {
    const f = fileInput.files[0];
    if (f && f.type && f.type.startsWith('image/')) {
      selectedFile = f;
      btnPredictImage.disabled = false;
      imageError.textContent = '';
      updatePreview(f);
    }
  });

  function showResult(data) {
    const finalLabel = data.final_label || data.label;
    const label = finalLabel || 'uncertain';
    const confidence = typeof data.confidence === 'number' ? data.confidence : 0;
    const textPreview = data.text_preview || '';

    resultCard.classList.remove('phishing', 'legitimate', 'uncertain', 'not_email');
    resultCard.classList.add(label);
    resultCard.classList.add('visible');

    let labelText;
    if (label === 'phishing') {
      labelText = 'Phishing email';
    } else if (label === 'legitimate') {
      labelText = 'Legitimate email';
    } else if (label === 'not_email') {
      labelText = 'Not an email';
    } else {
      labelText = 'Uncertain result';
    }
    resultLabel.textContent = labelText;

    if (typeof data.phishing_probability === 'number' && typeof data.legitimate_probability === 'number') {
      const pPhish = Math.round(data.phishing_probability * 100);
      const pLegit = Math.round(data.legitimate_probability * 100);
      const reason = data.reason ? String(data.reason).replace(/_/g, ' ') : '';
      resultConfidence.textContent = reason
        ? `Model confidence: ${Math.round(confidence * 100)}% · ${reason}`
        : `Model confidence: ${Math.round(confidence * 100)}%`;

      if (resultProbChip) {
        resultProbChip.textContent = `Phish ${pPhish}% · Legit ${pLegit}%`;
      }
    } else {
      resultConfidence.textContent = 'Model confidence: ' + (Math.round(confidence * 100)) + '%';
      if (resultProbChip) {
        resultProbChip.textContent = '';
      }
    }

    if (resultScoreChip) {
      if (typeof data.email_score === 'number') {
        const score = Math.round(data.email_score * 1000) / 1000;
        resultScoreChip.textContent = `Email‑likeness: ${score}`;
      } else {
        resultScoreChip.textContent = '';
      }
    }

    // Contextual message
    let msg = '';
    if (label === 'phishing') {
      msg = 'Treat this email as dangerous. Do not click links or download attachments unless you are absolutely sure.';
    } else if (label === 'legitimate') {
      msg = 'The model considers this email legitimate, but you should still exercise normal caution.';
    } else if (label === 'uncertain') {
      msg = 'We are not confident about this result. Please review the email manually before taking any action.';
    } else if (label === 'not_email') {
      msg = 'This image does not look like an email. The prediction may not be meaningful.';
    }
    if (resultMessage) {
      resultMessage.textContent = msg;
    }

    resultPreview.textContent = textPreview || '';
  }

  function showError(el, msg) {
    el.textContent = msg;
  }

  btnPredictImage.addEventListener('click', async function () {
    if (!selectedFile) return;
    imageError.textContent = '';
    btnPredictImage.disabled = true;
    startProgress('image');
    const form = new FormData();
    form.append('file', selectedFile);
    try {
      const r = await fetch('/predict/image', { method: 'POST', body: form });
      const data = await r.json();
      if (!r.ok) throw new Error(data.detail || r.statusText);
      showResult(data);
    } catch (e) {
      showError(imageError, e.message || 'Request failed');
    } finally {
      btnPredictImage.disabled = false;
      finishProgress();
    }
  });

  btnPredictText.addEventListener('click', async function () {
    textError.textContent = '';
    const subject = (subjectInput.value || '').trim();
    const body = (bodyInput.value || '').trim();
    if (!subject && !body) {
      showError(textError, 'Enter subject and/or body.');
      return;
    }
    btnPredictText.disabled = true;
    startProgress('text');
    try {
      const r = await fetch('/predict/text', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ subject: subject || undefined, body: body || undefined }),
      });
      const data = await r.json();
      if (!r.ok) throw new Error(data.detail || r.statusText);
      showResult(data);
    } catch (e) {
      showError(textError, e.message || 'Request failed');
    } finally {
      btnPredictText.disabled = false;
      finishProgress();
    }
  });
})();
