// メイン JavaScript ファイル

// DOM読み込み完了後の初期化
document.addEventListener('DOMContentLoaded', function() {
    initializeSliders();
    initializeFormValidation();
    initializeTooltips();
    initializeProgressBar();
});

// スライダー初期化
function initializeSliders() {
    const sliders = document.querySelectorAll('input[type="range"]');
    
    sliders.forEach(slider => {
        const valueDisplay = document.getElementById(slider.id + '-value');
        
        // 初期値表示
        if (valueDisplay) {
            valueDisplay.textContent = slider.value;
        }
        
        // スライダー変更時のイベント
        slider.addEventListener('input', function() {
            if (valueDisplay) {
                valueDisplay.textContent = this.value;
            }
            
            // 重み付けスライダーの場合、合計を更新
            if (this.classList.contains('weight-slider')) {
                updateWeightTotal();
            }
        });
        
        // スライダーのアクセシビリティ向上
        slider.addEventListener('keydown', function(e) {
            let step = parseInt(this.step) || 1;
            let currentValue = parseInt(this.value);
            
            switch(e.key) {
                case 'ArrowLeft':
                case 'ArrowDown':
                    e.preventDefault();
                    this.value = Math.max(parseInt(this.min), currentValue - step);
                    this.dispatchEvent(new Event('input'));
                    break;
                case 'ArrowRight':
                case 'ArrowUp':
                    e.preventDefault();
                    this.value = Math.min(parseInt(this.max), currentValue + step);
                    this.dispatchEvent(new Event('input'));
                    break;
                case 'Home':
                    e.preventDefault();
                    this.value = this.min;
                    this.dispatchEvent(new Event('input'));
                    break;
                case 'End':
                    e.preventDefault();
                    this.value = this.max;
                    this.dispatchEvent(new Event('input'));
                    break;
            }
        });
    });
}

// 重み付けスライダーの合計更新
function updateWeightTotal() {
    const weightSliders = document.querySelectorAll('.weight-slider');
    const totalDisplay = document.getElementById('total-weight');
    
    if (!totalDisplay) return;
    
    let total = 0;
    weightSliders.forEach(slider => {
        total += parseInt(slider.value) || 0;
    });
    
    totalDisplay.textContent = total;
    
    // 合計に応じてスタイル変更
    totalDisplay.className = total === 100 ? 'correct' : 'incorrect';
    
    // 警告表示
    const existingWarning = document.querySelector('.weight-warning');
    if (existingWarning) {
        existingWarning.remove();
    }
    
    if (total !== 100) {
        const warning = document.createElement('div');
        warning.className = 'weight-warning';
        warning.style.cssText = 'color: #dc3545; font-size: 0.9em; text-align: center; margin-top: 10px;';
        warning.textContent = `合計を100%に調整してください（現在: ${total}%）`;
        totalDisplay.parentNode.appendChild(warning);
    }
}

// 重み付け自動調整機能
function autoAdjustWeights() {
    const weightSliders = document.querySelectorAll('.weight-slider');
    if (weightSliders.length === 0) return;
    
    const values = Array.from(weightSliders).map(slider => parseInt(slider.value));
    const currentTotal = values.reduce((sum, val) => sum + val, 0);
    
    if (currentTotal === 100) return;
    
    const target = 100;
    const difference = target - currentTotal;
    const adjustment = Math.floor(difference / weightSliders.length);
    const remainder = difference % weightSliders.length;
    
    weightSliders.forEach((slider, index) => {
        let newValue = parseInt(slider.value) + adjustment;
        if (index < remainder) {
            newValue += 1;
        }
        
        // 範囲内に収める
        newValue = Math.max(0, Math.min(100, newValue));
        slider.value = newValue;
        
        // 表示を更新
        const valueDisplay = document.getElementById(slider.id + '-value');
        if (valueDisplay) {
            valueDisplay.textContent = newValue;
        }
    });
    
    updateWeightTotal();
}

// フォームバリデーション初期化
function initializeFormValidation() {
    const forms = document.querySelectorAll('form');
    
    forms.forEach(form => {
        // 質問紙フォームはテンプレート側で独自検証を実行するため除外
        if (form.id === 'questionnaireForm') {
            return;
        }
        form.addEventListener('submit', function(e) {
            if (!validateForm(this)) {
                e.preventDefault();
                showValidationErrors();
            }
        });
        
        // リアルタイムバリデーション
        const inputs = form.querySelectorAll('input, textarea, select');
        inputs.forEach(input => {
            input.addEventListener('blur', function() {
                validateField(this);
            });
        });
    });
}

// フォームバリデーション
function validateForm(form) {
    let isValid = true;
    const errors = [];
    
    // 必須フィールドチェック
    const requiredFields = form.querySelectorAll('[required]');
    requiredFields.forEach(field => {
        if (!field.value.trim()) {
            isValid = false;
            errors.push(`${getFieldLabel(field)}は必須です。`);
            markFieldError(field);
        } else {
            clearFieldError(field);
        }
    });
    
    // 重み付け合計チェック（該当フォームの場合）
    const totalWeight = document.getElementById('total-weight');
    if (totalWeight) {
        const total = parseInt(totalWeight.textContent);
        if (total !== 100) {
            isValid = false;
            errors.push('重要度の合計が100%になるよう調整してください。');
        }
    }
    
    // ラジオボタンチェック
    const radioGroups = {};
    const radioButtons = form.querySelectorAll('input[type="radio"][required]');
    radioButtons.forEach(radio => {
        if (!radioGroups[radio.name]) {
            radioGroups[radio.name] = [];
        }
        radioGroups[radio.name].push(radio);
    });
    
    Object.keys(radioGroups).forEach(groupName => {
        const group = radioGroups[groupName];
        const isChecked = group.some(radio => radio.checked);
        if (!isChecked) {
            isValid = false;
            errors.push(`${getFieldLabel(group[0])}を選択してください。`);
        }
    });
    
    // エラー表示
    if (!isValid) {
        displayErrors(errors);
    }
    
    return isValid;
}

// フィールドバリデーション
function validateField(field) {
    clearFieldError(field);
    
    if (field.required && !field.value.trim()) {
        markFieldError(field, `${getFieldLabel(field)}は必須です。`);
        return false;
    }
    
    return true;
}

// フィールドラベル取得
function getFieldLabel(field) {
    const label = document.querySelector(`label[for="${field.id}"]`);
    if (label) {
        return label.textContent.replace('*', '').trim();
    }
    return field.name || field.id || 'このフィールド';
}

// フィールドエラー表示
function markFieldError(field, message) {
    field.classList.add('error');
    
    // 既存のエラーメッセージを削除
    const existingError = field.parentNode.querySelector('.field-error');
    if (existingError) {
        existingError.remove();
    }
    
    // 新しいエラーメッセージを追加
    if (message) {
        const errorElement = document.createElement('div');
        errorElement.className = 'field-error';
        errorElement.style.cssText = 'color: #dc3545; font-size: 0.8em; margin-top: 5px;';
        errorElement.textContent = message;
        field.parentNode.appendChild(errorElement);
    }
}

// フィールドエラークリア
function clearFieldError(field) {
    field.classList.remove('error');
    const errorElement = field.parentNode.querySelector('.field-error');
    if (errorElement) {
        errorElement.remove();
    }
}

// エラー一覧表示
function displayErrors(errors) {
    // 既存のエラー表示を削除
    const existingErrorBox = document.querySelector('.error-box');
    if (existingErrorBox) {
        existingErrorBox.remove();
    }
    
    if (errors.length === 0) return;
    
    const errorBox = document.createElement('div');
    errorBox.className = 'error-box';
    errorBox.style.cssText = `
        background: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        padding: 15px;
        border-radius: 6px;
        margin: 20px 0;
    `;
    
    const errorTitle = document.createElement('h4');
    errorTitle.textContent = '入力内容を確認してください';
    errorTitle.style.margin = '0 0 10px 0';
    errorBox.appendChild(errorTitle);
    
    const errorList = document.createElement('ul');
    errorList.style.margin = '0';
    errors.forEach(error => {
        const errorItem = document.createElement('li');
        errorItem.textContent = error;
        errorList.appendChild(errorItem);
    });
    
    errorBox.appendChild(errorList);
    
    // フォームの最初に挿入
    const form = document.querySelector('form');
    if (form) {
        form.insertBefore(errorBox, form.firstChild);
        errorBox.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
}

// バリデーションエラー表示
function showValidationErrors() {
    const errorBox = document.querySelector('.error-box');
    if (errorBox) {
        errorBox.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
}

// ツールチップ初期化
function initializeTooltips() {
    const tooltipElements = document.querySelectorAll('[data-tooltip]');
    
    tooltipElements.forEach(element => {
        element.addEventListener('mouseenter', showTooltip);
        element.addEventListener('mouseleave', hideTooltip);
        element.addEventListener('focus', showTooltip);
        element.addEventListener('blur', hideTooltip);
    });
}

// ツールチップ表示
function showTooltip(e) {
    const element = e.target;
    const tooltipText = element.getAttribute('data-tooltip');
    
    if (!tooltipText) return;
    
    const tooltip = document.createElement('div');
    tooltip.className = 'tooltip';
    tooltip.textContent = tooltipText;
    tooltip.style.cssText = `
        position: absolute;
        background: #333;
        color: white;
        padding: 8px 12px;
        border-radius: 4px;
        font-size: 0.8em;
        z-index: 1000;
        pointer-events: none;
        opacity: 0;
        transition: opacity 0.3s;
    `;
    
    document.body.appendChild(tooltip);
    
    // 位置調整
    const rect = element.getBoundingClientRect();
    tooltip.style.left = rect.left + (rect.width / 2) - (tooltip.offsetWidth / 2) + 'px';
    tooltip.style.top = rect.top - tooltip.offsetHeight - 8 + 'px';
    
    // フェードイン
    setTimeout(() => {
        tooltip.style.opacity = '1';
    }, 10);
    
    element._tooltip = tooltip;
}

// ツールチップ非表示
function hideTooltip(e) {
    const element = e.target;
    if (element._tooltip) {
        element._tooltip.remove();
        delete element._tooltip;
    }
}

// プログレスバー初期化
function initializeProgressBar() {
    const progressBar = document.querySelector('.progress-bar');
    if (!progressBar) return;
    
    // URLから進行状況を判定
    const path = window.location.pathname;
    let progress = 0;
    
    if (path.includes('questionnaire')) {
        progress = path.includes('pre') ? 10 : 90;
    } else if (path.includes('decision')) {
        progress = 30;
    } else if (path.includes('aggregation')) {
        progress = 50;
    } else if (path.includes('intervention')) {
        progress = 60;
    } else if (path.includes('final_decision')) {
        progress = 80;
    } else if (path.includes('complete')) {
        progress = 100;
    }
    
    updateProgressBar(progress);
}

// プログレスバー更新
function updateProgressBar(percentage) {
    const progressBar = document.querySelector('.progress-bar');
    const progressFill = document.querySelector('.progress-fill');
    
    if (progressBar && progressFill) {
        progressFill.style.width = percentage + '%';
        progressFill.setAttribute('aria-valuenow', percentage);
    }
}

// ローディング表示
function showLoading(message = '処理中...') {
    const loading = document.createElement('div');
    loading.id = 'loading-overlay';
    loading.style.cssText = `
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(0, 0, 0, 0.5);
        display: flex;
        justify-content: center;
        align-items: center;
        z-index: 9999;
    `;
    
    const spinner = document.createElement('div');
    spinner.style.cssText = `
        background: white;
        padding: 30px;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    `;
    
    const spinnerIcon = document.createElement('div');
    spinnerIcon.style.cssText = `
        width: 40px;
        height: 40px;
        border: 4px solid #f3f3f3;
        border-top: 4px solid #667eea;
        border-radius: 50%;
        animation: spin 1s linear infinite;
        margin: 0 auto 15px;
    `;
    
    const spinnerText = document.createElement('div');
    spinnerText.textContent = message;
    spinnerText.style.color = '#333';
    
    spinner.appendChild(spinnerIcon);
    spinner.appendChild(spinnerText);
    loading.appendChild(spinner);
    
    // CSS アニメーション追加
    if (!document.querySelector('#spinner-styles')) {
        const style = document.createElement('style');
        style.id = 'spinner-styles';
        style.textContent = `
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
        `;
        document.head.appendChild(style);
    }
    
    document.body.appendChild(loading);
}

// ローディング非表示
function hideLoading() {
    const loading = document.getElementById('loading-overlay');
    if (loading) {
        loading.remove();
    }
}

// ユーティリティ関数：Ajax リクエスト
function sendRequest(url, data, method = 'POST') {
    showLoading();
    
    return fetch(url, {
        method: method,
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(data)
    })
    .then(response => {
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        return response.json();
    })
    .then(result => {
        hideLoading();
        return result;
    })
    .catch(error => {
        hideLoading();
        console.error('Request error:', error);
        alert('エラーが発生しました。もう一度お試しください。');
        throw error;
    });
}

// ページ遷移
function navigateToPage(url) {
    showLoading('ページを読み込んでいます...');
    window.location.href = url;
}

// セッション状態チェック
function checkSessionStatus() {
    return fetch('/api/session-status')
        .then(response => response.json())
        .then(data => {
            if (!data.valid) {
                alert('セッションが無効です。最初からやり直してください。');
                window.location.href = '/';
            }
            return data;
        })
        .catch(error => {
            console.error('Session check error:', error);
        });
}

// キーボードショートカット
document.addEventListener('keydown', function(e) {
    // Ctrl+Enter でフォーム送信
    if (e.ctrlKey && e.key === 'Enter') {
        const form = document.querySelector('form');
        if (form) {
            form.dispatchEvent(new Event('submit'));
        }
    }
    
    // Ctrl+Shift+A で重み付け自動調整
    if (e.ctrlKey && e.shiftKey && e.key === 'A') {
        e.preventDefault();
        autoAdjustWeights();
    }
});

// ページ離脱警告（フォーム入力中）
// グローバルフラグで制御（index等から無効化可能）
if (typeof window.__formModified === 'undefined') window.__formModified = false;
if (typeof window.__suppressBeforeUnload === 'undefined') window.__suppressBeforeUnload = false;

document.addEventListener('input', function() {
    window.__formModified = true;
});

document.addEventListener('submit', function() {
    window.__formModified = false;
});

window.addEventListener('beforeunload', function(e) {
    if (window.__suppressBeforeUnload) return;
    if (window.__formModified) {
        e.preventDefault();
        e.returnValue = '入力中のデータが失われます。本当にページを離れますか？';
    }
});

// エクスポート（必要に応じて）
window.ExperimentUtils = {
    updateWeightTotal,
    autoAdjustWeights,
    validateForm,
    showLoading,
    hideLoading,
    sendRequest,
    navigateToPage,
    checkSessionStatus
};
