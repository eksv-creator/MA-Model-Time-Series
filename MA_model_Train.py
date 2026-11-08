import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.regression.linear_model import OLS
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from scipy.stats import jarque_bera
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.stats.diagnostic import het_white
import statsmodels.api as sm
from scipy.optimize import minimize
from scipy.stats import norm
import warnings

warnings.filterwarnings('ignore')

# ========================== ЗАГРУЗКА ДАННЫХ ==========================
df = pd.read_csv("eminar.csv", sep=";", parse_dates=['Data'], dayfirst=True)
df['Data'] = pd.to_datetime(df['Data'], format='%d.%m.%Y')
print(f"Всего строк в файле: {len(df)}")
df = df.dropna()
print(f"После удаления пропусков: {len(df)}")
numeric_cols = df.columns.drop('Data')
print(f"Число наблюдений: {len(df)}")

# Визуализация исходного ряда
for col in numeric_cols:
    plt.figure(figsize=(10, 4))
    plt.plot(df['Data'], df[col], linewidth=1.2)
    plt.title(f' {col}', fontsize=14)
    plt.xlabel('Дата')
    plt.ylabel('Значение')
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.show()

# ========================== 1. НОРМАЛЬНОСТЬ ИСХОДНЫХ ДАННЫХ ==========================
for col in numeric_cols:
    jb_stat, jb_p = stats.jarque_bera(df[col].dropna())
    norm_type = "не нормальное" if jb_p < 0.05 else "нормальное"
    print(f"• {col}: p = {jb_p:.4f} → {norm_type}")

# ========================== 2. РАЗБИЕНИЕ НА ОБУЧАЮЩУЮ И ТЕСТОВУЮ ВЫБОРКИ ==========================
y = df['Share price'].copy()
Shag = 10
train = y.iloc[:-Shag]
test = y.iloc[-Shag:]
print(f"\nОбучающая выборка: {len(train)} наблюдений")
print(f"Тестовая выборка: {len(test)} наблюдений")
print("Фактические значения test:", test.values)

# Визуализация разделения
plt.figure(figsize=(12, 5))
plt.plot(range(len(train)), train, label='Обучающая выборка')
plt.plot(range(len(train), len(train) + Shag), test, 'o-', label='Тестовая выборка')
plt.title('Разделение ряда на train и test')
plt.legend()
plt.grid(True)
plt.show()


# ========================== 3. СТАЦИОНАРНОСТЬ НА ОБУЧАЮЩЕЙ ВЫБОРКЕ ==========================
def adf_manual(y, maxlag=5, with_trend=False):
    y = np.array(y)
    n = len(y)
    dy = np.diff(y)
    p = maxlag
    y_dep = dy[p:]
    X = np.ones((len(y_dep), 1))
    y_lag = y[p:-1]
    X = np.column_stack((X, y_lag))
    for lag in range(1, p + 1):
        lagged_dy = dy[p - lag: n - 1 - lag]
        X = np.column_stack((X, lagged_dy))
    if with_trend:
        trend = np.arange(p + 1, n)
        X = np.column_stack((X, trend))
    model = OLS(y_dep, X).fit()
    tstat = model.tvalues[1]
    if with_trend:
        crit = {'1%': -3.96, '5%': -3.41, '10%': -3.12}
    else:
        crit = {'1%': -3.43, '5%': -2.86, '10%': -2.57}
    return tstat, crit


print("\n=== ADF-тест на обучающей выборке ===")
adf_result = adfuller(train, regression='c')
print(f"Встроенный ADF: statistic = {adf_result[0]:.4f}, p-value = {adf_result[1]:.4f}")
if adf_result[1] > 0.05:
    print("→ Ряд НЕ стационарен. Берём первую разность.")
    train_diff = train.diff().dropna()
    print(f"Длина разности: {len(train_diff)}")
    adf_diff = adfuller(train_diff, regression='c')
    print(f"ADF для разности: p-value = {adf_diff[1]:.4f} → стационарна")
    y_for_model = train_diff
    used_diff = True
else:
    print("→ Ряд стационарен.")
    train_diff = train
    y_for_model = train
    used_diff = False
plt.figure(figsize=(12, 5))
plt.plot(train_diff.index, train_diff.values, linewidth=1.2)
plt.title('Ряд после взятия первой разности')
plt.xlabel('Индекс')
plt.ylabel('Δy')
plt.grid(True, alpha=0.4)
plt.tight_layout()
plt.show()
# Ручной ADF для проверки
tstat, crit = adf_manual(train, maxlag=5, with_trend=False)
print(f"\nРучной ADF: t-статистика = {tstat:.4f}")
print("Критические значения:", crit)
if tstat < crit['5%']:
    print("→ Ряд стационарен (ручной)")
else:
    print("→ Ряд НЕ стационарен (ручной)")


# ========================== 4. ACF И PACF СТАЦИОНАРНОГО РЯДА (НА ОБУЧАЮЩЕЙ ВЫБОРКЕ) ==========================
def acf_manual(x, nlags=30):
    x = np.array(x)
    n = len(x)
    x_mean = np.mean(x)
    denom = np.sum((x - x_mean) ** 2)
    acf_vals = []
    for k in range(0, nlags + 1):
        if k == 0:
            acf_vals.append(1.0)
        else:
            num = np.sum((x[k:] - x_mean) * (x[:-k] - x_mean))
            acf_vals.append(num / denom)
    return np.array(acf_vals)


plt.figure(figsize=(12, 4))
plot_acf(y_for_model, lags=30, alpha=0.05)
plt.title('ACF стационарного ряда (обучающая выборка)')
plt.show()
fig, ax = plt.subplots(figsize=(12, 4))
plot_acf(y_for_model, lags=30, alpha=0.05, ax=ax)
ax.set_title('ACF стационарного ряда (обучающая выборка)')
fig.savefig('acf_train.png', dpi=150)
plt.close()

plt.figure(figsize=(12, 4))
plot_pacf(y_for_model, lags=30, method='ywm', alpha=0.05)
plt.title('PACF стационарного ряда (обучающая выборка)')
plt.show()
fig, ax = plt.subplots(figsize=(12, 4))
plot_pacf(y_for_model, lags=30, method='ywm', alpha=0.05, ax=ax)
ax.set_title('PACF стационарного ряда (обучающая выборка)')
fig.savefig('pacf_train.png', dpi=150)
plt.close()

acf_vals = acf_manual(y_for_model, nlags=30)
n = len(y_for_model)
ci = 1.96 / np.sqrt(n)
print("\nРучная ACF (первые 10 лагов):")
for i in range(1, 11):
    print(f"  лаг {i}: {acf_vals[i]:.4f} {'*' if abs(acf_vals[i]) > ci else ''}")
print(f"95% доверительная граница: ±{ci:.4f}")
sig_lags = np.where(np.abs(acf_vals[1:]) > ci)[0] + 1
if len(sig_lags) > 0:
    print(f"Значимые лаги: {sig_lags}")
    consecutive_lags = []
    for lag in sig_lags:
        if lag == len(consecutive_lags) + 1:
            consecutive_lags.append(lag)
        else:
            break
    if len(consecutive_lags) > 0:
        suggested_q = consecutive_lags[-1]
        print(f"Предполагаемый q по раннему обрыву ACF: {suggested_q}")
    else:
        suggested_q = 0
        print("По ACF нет чёткого раннего обрыва, порядок уточняем по AIC.")
else:
    suggested_q = 0
    print("Значимых лагов нет.")

# ========================== 5. ВЫБОР ПОРЯДКА q ПО AIC (НА ОБУЧАЮЩЕЙ ВЫБОРКЕ) ==========================
print("\n=== Выбор порядка q по AIC (только train) ===")
best_q, best_aic = 0, np.inf
for q in range(0, 9):
    try:
        if used_diff:
            model = ARIMA(train, order=(0, 1, q)).fit(method_kwargs={'disp': False})
        else:
            model = ARIMA(train, order=(0, 0, q)).fit(method_kwargs={'disp': False})
        aic = model.aic
        bic = model.bic
        print(f"q={q}: AIC={aic:.2f}, BIC={bic:.2f}")
        if aic < best_aic:
            best_aic = aic
            best_q = q
    except:
        print(f"q={q}: ошибка при оценке")
print(f"\nОптимальный q по AIC: {best_q} (AIC = {best_aic:.2f})")

# ========================== 6. ВЫБОР ПОРЯДКА p ПО AIC (НА ОБУЧАЮЩЕЙ ВЫБОРКЕ) ==========================
print("\n=== Выбор порядка p по AIC (только train) ===")
best_p, best_aic_ar = 0, np.inf
for p in range(0, 9):
    try:
        if used_diff:
            model = ARIMA(train, order=(p, 1, 0)).fit(method_kwargs={'disp': False})
        else:
            model = ARIMA(train, order=(p, 0, 0)).fit(method_kwargs={'disp': False})
        aic = model.aic
        bic = model.bic
        print(f"p={p}: AIC={aic:.2f}, BIC={bic:.2f}")
        if aic < best_aic_ar:
            best_aic_ar = aic
            best_p = p
    except:
        print(f"p={p}: ошибка при оценке")
print(f"\nОптимальный p по AIC: {best_p} (AIC = {best_aic_ar:.2f})")

# ========================== 7. ВЫБОР ПОРЯДКОВ (p, q) ДЛЯ ARMA ПО AIC ==========================
print("\n=== Выбор порядков (p, q) для ARMA по AIC (только train) ===")
best_p_arma, best_q_arma, best_aic_arma = 0, 0, np.inf
for p in range(0, 5):
    for q in range(0, 5):
        if p == 0 and q == 0:
            continue
        try:
            if used_diff:
                model = ARIMA(train, order=(p, 1, q)).fit(method_kwargs={'disp': False})
            else:
                model = ARIMA(train, order=(p, 0, q)).fit(method_kwargs={'disp': False})
            aic = model.aic
            bic = model.bic
            print(f"p={p}, q={q}: AIC={aic:.2f}, BIC={bic:.2f}")
            if aic < best_aic_arma:
                best_aic_arma = aic
                best_p_arma = p
                best_q_arma = q
        except:
            print(f"p={p}, q={q}: ошибка при оценке")
print(f"\nОптимальные p и q для ARMA по AIC: p={best_p_arma}, q={best_q_arma} (AIC = {best_aic_arma:.2f})")


# ========================== Для выводы уравнений ==========================
def format_signed_term(value, term):
    sign = "+" if value >= 0 else "-"
    return f" {sign} {abs(value):.6f}·{term}"

def build_model_equation(model, used_diff):
    lhs = "Δy_t" if used_diff else "y_t"
    params = dict(zip(model.param_names, model.params))

    const = 0.0
    if 'const' in params:
        const = params['const']
    elif 'intercept' in params:
        const = params['intercept']

    ar_terms = []
    ma_terms = []

    for name, value in params.items():
        if name.startswith('ar.L'):
            lag = int(name.split('L')[1])
            ar_terms.append((lag, value))
        elif name.startswith('ma.L'):
            lag = int(name.split('L')[1])
            ma_terms.append((lag, value))

    ar_terms.sort(key=lambda x: x[0])
    ma_terms.sort(key=lambda x: x[0])

    equation = f"{lhs} = {const:.6f}"

    for lag, value in ar_terms:
        rhs = f"Δy_(t-{lag})" if used_diff else f"y_(t-{lag})"
        equation += format_signed_term(value, rhs)

    equation += " + ε_t"

    for lag, value in ma_terms:
        equation += format_signed_term(value, f"ε_(t-{lag})")

    return equation
# ========================== 8. ОЦЕНКА МОДЕЛИ MA(q) ВСТРОЕННЫМ ARIMA ==========================
if used_diff:
    model_ma = ARIMA(train, order=(0, 1, best_q)).fit()
else:
    model_ma = ARIMA(train, order=(0, 0, best_q)).fit()
print("\n=== Оценённая модель MA(q) (встроенный ARIMA) ===")
print(model_ma.summary())

print("\nЗначимость параметров (встроенная, MA):")
for name, param, pval in zip(model_ma.param_names, model_ma.params, model_ma.pvalues):
    print(f"  {name}: {param:.6f} (p={pval:.4f}) {'значим' if pval < 0.05 else 'незначим'}")

# ========================== 9. ОЦЕНКА МОДЕЛИ AR(p) ВСТРОЕННЫМ ARIMA ==========================
if used_diff:
    model_ar = ARIMA(train, order=(best_p, 1, 0)).fit()
else:
    model_ar = ARIMA(train, order=(best_p, 0, 0)).fit()
print("\n=== Оценённая модель AR(p) (встроенный ARIMA) ===")
print(model_ar.summary())

print("\nЗначимость параметров (встроенная, AR):")
for name, param, pval in zip(model_ar.param_names, model_ar.params, model_ar.pvalues):
    print(f"  {name}: {param:.6f} (p={pval:.4f}) {'значим' if pval < 0.05 else 'незначим'}")

# ========================== 10. ОЦЕНКА МОДЕЛИ ARMA(p, q) ВСТРОЕННЫМ ARIMA ==========================
if used_diff:
    model_arma = ARIMA(train, order=(best_p_arma, 1, best_q_arma)).fit()
else:
    model_arma = ARIMA(train, order=(best_p_arma, 0, best_q_arma)).fit()
print("\n=== Оценённая модель ARMA(p, q) (встроенный ARIMA) ===")
print(model_arma.summary())

print("\nЗначимость параметров (встроенная, ARMA):")
for name, param, pval in zip(model_arma.param_names, model_arma.params, model_arma.pvalues):
    print(f"  {name}: {param:.6f} (p={pval:.4f}) {'значим' if pval < 0.05 else 'незначим'}")


# ========================== 11. РУЧНАЯ ОЦЕНКА МОДЕЛИ MA(q) (MLE) ==========================
def estimate_ma_q_mle(y, q):
    y = np.asarray(y)
    n = len(y)
    mu0 = np.mean(y)
    theta0 = np.zeros(q)
    sigma0 = np.std(y)
    x0 = np.concatenate(([mu0], theta0, [sigma0]))

    def neg_log_likelihood(params):
        mu = params[0]
        thetas = params[1:1 + q]
        sigma = params[1 + q]
        if sigma <= 0:
            return 1e10
        eps = np.zeros(n)
        for t in range(n):
            if t == 0:
                eps[t] = y[t] - mu
            else:
                sum_theta = 0.0
                for j in range(1, min(t, q) + 1):
                    sum_theta += thetas[j - 1] * eps[t - j]
                eps[t] = y[t] - mu - sum_theta
        ll = -0.5 * n * np.log(2 * np.pi) - n * np.log(sigma) - 0.5 * np.sum(eps ** 2) / sigma ** 2
        return -ll

    res = minimize(neg_log_likelihood, x0, method='L-BFGS-B', options={'maxiter': 1000})
    if not res.success:
        print("Предупреждение: оптимизация не сошлась", res.message)
    params = res.x
    mu = params[0]
    thetas = params[1:1 + q]
    sigma = params[1 + q]

    eps_hess = 1e-5
    hessian = np.zeros((len(params), len(params)))
    for i in range(len(params)):
        for j in range(len(params)):
            params_ij = params.copy()
            params_ij[i] += eps_hess
            params_ij[j] += eps_hess
            f_ij = neg_log_likelihood(params_ij)
            params_i = params.copy()
            params_i[i] += eps_hess
            f_i = neg_log_likelihood(params_i)
            params_j = params.copy()
            params_j[j] += eps_hess
            f_j = neg_log_likelihood(params_j)
            f = neg_log_likelihood(params)
            hessian[i, j] = (f_ij - f_i - f_j + f) / (eps_hess ** 2)
    try:
        cov = np.linalg.inv(hessian)
        std_errors = np.sqrt(np.diag(cov))
    except:
        std_errors = np.full(len(params), np.nan)
    z_stats = params / std_errors
    p_values = 2 * (1 - norm.cdf(np.abs(z_stats)))
    return {
        'mu': mu, 'thetas': thetas, 'sigma': sigma,
        'std_errors': std_errors, 'z_stats': z_stats, 'p_values': p_values,
        'logL': -res.fun, 'n': n, 'q': q
    }

def estimate_ar_p_mle(y, p):
    y = np.asarray(y)
    n = len(y)
    mu0 = np.mean(y)
    phi0 = np.zeros(p)
    sigma0 = np.std(y)
    x0 = np.concatenate(([mu0], phi0, [sigma0]))

    def neg_log_likelihood(params):
        mu = params[0]
        phis = params[1:1 + p]
        sigma = params[1 + p]
        if sigma <= 0:
            return 1e10
        eps = np.zeros(n)
        for t in range(n):
            if t < p:
                eps[t] = y[t] - mu
            else:
                ar_sum = 0.0
                for j in range(1, p + 1):
                    ar_sum += phis[j - 1] * (y[t - j] - mu)
                eps[t] = y[t] - mu - ar_sum
        ll = -0.5 * n * np.log(2 * np.pi) - n * np.log(sigma) - 0.5 * np.sum(eps[p:] ** 2) / sigma ** 2
        return -ll

    res = minimize(neg_log_likelihood, x0, method='L-BFGS-B', options={'maxiter': 1000})
    if not res.success:
        print("Предупреждение: оптимизация не сошлась", res.message)
    params = res.x
    mu = params[0]
    phis = params[1:1 + p]
    sigma = params[1 + p]

    eps_hess = 1e-5
    hessian = np.zeros((len(params), len(params)))
    for i in range(len(params)):
        for j in range(len(params)):
            params_ij = params.copy()
            params_ij[i] += eps_hess
            params_ij[j] += eps_hess
            f_ij = neg_log_likelihood(params_ij)
            params_i = params.copy()
            params_i[i] += eps_hess
            f_i = neg_log_likelihood(params_i)
            params_j = params.copy()
            params_j[j] += eps_hess
            f_j = neg_log_likelihood(params_j)
            f = neg_log_likelihood(params)
            hessian[i, j] = (f_ij - f_i - f_j + f) / (eps_hess ** 2)
    try:
        cov = np.linalg.inv(hessian)
        std_errors = np.sqrt(np.diag(cov))
    except:
        std_errors = np.full(len(params), np.nan)
    z_stats = params / std_errors
    p_values = 2 * (1 - norm.cdf(np.abs(z_stats)))
    return {
        'mu': mu, 'phis': phis, 'sigma': sigma,
        'std_errors': std_errors, 'z_stats': z_stats, 'p_values': p_values,
        'logL': -res.fun, 'n': n, 'p': p
    }

def estimate_arma_pq_mle(y, p, q):
    y = np.asarray(y, dtype=float)
    n = len(y)

    try:
        start_model = ARIMA(y, order=(p, 0, q)).fit()
        mu0 = float(np.mean(y))
        phis0 = np.zeros(p)
        thetas0 = np.zeros(q)
        sigma0 = float(np.std(y, ddof=1))

        for i, name in enumerate(start_model.param_names):
            if name.startswith('ar.L'):
                lag = int(name.split('L')[1]) - 1
                if lag < p:
                    phis0[lag] = start_model.params[i]
            elif name.startswith('ma.L'):
                lag = int(name.split('L')[1]) - 1
                if lag < q:
                    thetas0[lag] = start_model.params[i]
            elif name == 'sigma2':
                sigma0 = float(np.sqrt(abs(start_model.params[i])))

        x0 = np.concatenate(([mu0], phis0, thetas0, [sigma0]))
    except:
        mu0 = np.mean(y)
        phis0 = np.zeros(p)
        thetas0 = np.zeros(q)
        sigma0 = np.std(y, ddof=1)
        x0 = np.concatenate(([mu0], phis0, thetas0, [sigma0]))

    bounds = [(None, None)]
    bounds += [(-0.99, 0.99)] * p
    bounds += [(-0.99, 0.99)] * q
    bounds += [(1e-6, None)]

    def neg_log_likelihood(params):
        mu = params[0]
        phis = params[1:1 + p]
        thetas = params[1 + p:1 + p + q]
        sigma = params[-1]

        if sigma <= 0:
            return 1e12

        eps = np.zeros(n)

        for t in range(n):
            ar_sum = 0.0
            ma_sum = 0.0

            if p > 0:
                for j in range(1, min(t, p) + 1):
                    ar_sum += phis[j - 1] * (y[t - j] - mu)

            if q > 0:
                for j in range(1, min(t, q) + 1):
                    ma_sum += thetas[j - 1] * eps[t - j]

            eps[t] = y[t] - mu - ar_sum - ma_sum

        start = max(p, q)
        eff_eps = eps[start:] if start < n else eps
        ll = -0.5 * len(eff_eps) * np.log(2 * np.pi) - len(eff_eps) * np.log(sigma) - 0.5 * np.sum(eff_eps ** 2) / sigma ** 2
        return -ll

    res = minimize(
        neg_log_likelihood,
        x0,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 5000}
    )

    if not res.success:
        print("Предупреждение: оптимизация не сошлась", res.message)

    params = res.x
    mu = params[0]
    phis = params[1:1 + p]
    thetas = params[1 + p:1 + p + q]
    sigma = params[-1]

    eps_hess = 1e-4
    k = len(params)
    hessian = np.zeros((k, k))
    f0 = neg_log_likelihood(params)

    for i in range(k):
        for j in range(k):
            p_ij = params.copy()
            p_i1 = params.copy()
            p_i2 = params.copy()
            p_j1 = params.copy()
            p_j2 = params.copy()

            if i == j:
                p_i1[i] += eps_hess
                p_i2[i] -= eps_hess
                f1 = neg_log_likelihood(p_i1)
                f2 = neg_log_likelihood(p_i2)
                hessian[i, i] = (f1 - 2 * f0 + f2) / (eps_hess ** 2)
            else:
                p_ij[i] += eps_hess
                p_ij[j] += eps_hess

                p_i1[i] += eps_hess
                p_j1[j] += eps_hess

                f_ij = neg_log_likelihood(p_ij)
                f_i = neg_log_likelihood(p_i1)
                f_j = neg_log_likelihood(p_j1)

                hessian[i, j] = (f_ij - f_i - f_j + f0) / (eps_hess ** 2)

    hessian = (hessian + hessian.T) / 2

    try:
        cov = np.linalg.pinv(hessian)
        std_errors = np.sqrt(np.maximum(np.diag(cov), 0))
        std_errors[~np.isfinite(std_errors)] = np.nan
    except:
        std_errors = np.full(k, np.nan)

    z_stats = np.full(k, np.nan)
    mask = (std_errors > 0) & np.isfinite(std_errors)
    z_stats[mask] = params[mask] / std_errors[mask]

    p_values = np.full(k, np.nan)
    mask_z = np.isfinite(z_stats)
    p_values[mask_z] = 2 * (1 - norm.cdf(np.abs(z_stats[mask_z])))

    return {
        'mu': mu,
        'phis': phis,
        'thetas': thetas,
        'sigma': sigma,
        'std_errors': std_errors,
        'z_stats': z_stats,
        'p_values': p_values,
        'logL': -res.fun,
        'n': n,
        'p': p,
        'q': q
    }


print("\n=== Ручная оценка MA(q) методом MLE ===")
if used_diff:
    print("Оценка проводится для стационарного ряда в первых разностях.")
else:
    print("Оценка проводится для исходного стационарного ряда.")
res_manual = estimate_ma_q_mle(y_for_model, best_q)

print(f"{'Параметр':<10} {'Коэфф.':<12} {'Ст.ошибка':<12} {'z-стат.':<10} {'p-value':<10}")
print("-" * 60)
print(f"{'mu':<10} {res_manual['mu']:<12.6f} {res_manual['std_errors'][0]:<12.6f} {res_manual['z_stats'][0]:<10.4f} {res_manual['p_values'][0]:<10.4f}")
for i in range(best_q):
    print(f"theta_{i + 1:<3} {res_manual['thetas'][i]:<12.6f} {res_manual['std_errors'][i + 1]:<12.6f} {res_manual['z_stats'][i + 1]:<10.4f} {res_manual['p_values'][i + 1]:<10.4f}")
print(f"{'sigma':<10} {res_manual['sigma']:<12.6f} {res_manual['std_errors'][-1]:<12.6f} {res_manual['z_stats'][-1]:<10.4f} {res_manual['p_values'][-1]:<10.4f}")
print(f"\nЛогарифмическое правдоподобие: {res_manual['logL']:.4f}")
print(f"AIC = {-2 * res_manual['logL'] + 2 * (best_q + 2):.2f}")
print(f"BIC = {-2 * res_manual['logL'] + (best_q + 2) * np.log(res_manual['n']):.2f}")

print("\n=== Ручная оценка AR(p) методом MLE ===")
res_manual_ar = estimate_ar_p_mle(y_for_model, best_p)

print(f"{'Параметр':<10} {'Коэфф.':<12} {'Ст.ошибка':<12} {'z-стат.':<10} {'p-value':<10}")
print("-" * 60)
print(f"{'mu':<10} {res_manual_ar['mu']:<12.6f} {res_manual_ar['std_errors'][0]:<12.6f} {res_manual_ar['z_stats'][0]:<10.4f} {res_manual_ar['p_values'][0]:<10.4f}")
for i in range(best_p):
    print(f"phi_{i + 1:<5} {res_manual_ar['phis'][i]:<12.6f} {res_manual_ar['std_errors'][i + 1]:<12.6f} {res_manual_ar['z_stats'][i + 1]:<10.4f} {res_manual_ar['p_values'][i + 1]:<10.4f}")
print(f"{'sigma':<10} {res_manual_ar['sigma']:<12.6f} {res_manual_ar['std_errors'][-1]:<12.6f} {res_manual_ar['z_stats'][-1]:<10.4f} {res_manual_ar['p_values'][-1]:<10.4f}")
print(f"\nЛогарифмическое правдоподобие: {res_manual_ar['logL']:.4f}")
print(f"AIC = {-2 * res_manual_ar['logL'] + 2 * (best_p + 2):.2f}")
print(f"BIC = {-2 * res_manual_ar['logL'] + (best_p + 2) * np.log(res_manual_ar['n']):.2f}")

print("\n=== Ручная оценка ARMA(p, q) методом MLE ===")
res_manual_arma = estimate_arma_pq_mle(y_for_model, best_p_arma, best_q_arma)

print(f"{'Параметр':<10} {'Коэфф.':<12} {'Ст.ошибка':<12} {'z-стат.':<10} {'p-value':<10}")
print("-" * 60)
print(f"{'mu':<10} {res_manual_arma['mu']:<12.6f} {res_manual_arma['std_errors'][0]:<12.6f} {res_manual_arma['z_stats'][0]:<10.4f} {res_manual_arma['p_values'][0]:<10.4f}")
for i in range(best_p_arma):
    print(f"phi_{i + 1:<5} {res_manual_arma['phis'][i]:<12.6f} {res_manual_arma['std_errors'][i + 1]:<12.6f} {res_manual_arma['z_stats'][i + 1]:<10.4f} {res_manual_arma['p_values'][i + 1]:<10.4f}")
for i in range(best_q_arma):
    print(f"theta_{i + 1:<3} {res_manual_arma['thetas'][i]:<12.6f} {res_manual_arma['std_errors'][1 + best_p_arma + i]:<12.6f} {res_manual_arma['z_stats'][1 + best_p_arma + i]:<10.4f} {res_manual_arma['p_values'][1 + best_p_arma + i]:<10.4f}")
print(f"{'sigma':<10} {res_manual_arma['sigma']:<12.6f} {res_manual_arma['std_errors'][-1]:<12.6f} {res_manual_arma['z_stats'][-1]:<10.4f} {res_manual_arma['p_values'][-1]:<10.4f}")
print(f"\nЛогарифмическое правдоподобие: {res_manual_arma['logL']:.4f}")
print(f"AIC = {-2 * res_manual_arma['logL'] + 2 * (best_p_arma + best_q_arma + 2):.2f}")
print(f"BIC = {-2 * res_manual_arma['logL'] + (best_p_arma + best_q_arma + 2) * np.log(res_manual_arma['n']):.2f}")

def residual_diagnostics(model_residuals, model_name):
    residuals = model_residuals

    print(f"\n=== Ljung-Box test для {model_name} (автокорреляция остатков) ===")
    lb_test = acorr_ljungbox(residuals, lags=[10, 20], return_df=True)
    print(lb_test)
    if (lb_test['lb_pvalue'].iloc[0] > 0.05) and (lb_test['lb_pvalue'].iloc[1] > 0.05):
        print("→ Остатки не коррелированы (модель адекватна)")
    else:
        print("→ Остатки коррелированы (модель требует улучшения)")

    arch_test = het_arch(residuals, maxlag=5)
    print(f"\n=== Тест Энгла (ARCH) для {model_name} ===")
    print(f"Статистика = {arch_test[0]:.4f}, p-value = {arch_test[1]:.4f}")
    if arch_test[1] < 0.05:
        print("→ Дисперсия остатков непостоянна (есть ARCH-эффекты)")
    else:
        print("→ Гетероскедастичности нет")

    print(f"\n=== Тест Уайта для {model_name} ===")
    if used_diff:
        residuals_white = np.asarray(residuals[1:])
        exog = sm.add_constant(np.asarray(y_for_model))
    else:
        residuals_white = np.asarray(residuals)
        exog = sm.add_constant(np.asarray(y_for_model))
    white_test = het_white(residuals_white, exog)
    print(f"White test LM-stat = {white_test[0]:.4f}, p-value = {white_test[1]:.4f}")
    if white_test[1] < 0.05:
        print("→ Отвергаем H0: гетероскедастичность присутствует")
    else:
        print("→ H0 не отвергается: гомоскедастичность")

    jb_stat, jb_p = jarque_bera(residuals)
    print(f"\n=== Jarque-Bera тест для {model_name} ===")
    print(f"Статистика = {jb_stat:.4f}, p-value = {jb_p:.4f}")
    if jb_p < 0.05:
        print("→ Остатки НЕ нормальны")
    else:
        print("→ Остатки нормальны")

# ========================== 12. ДИАГНОСТИКА ОСТАТКОВ ==========================
residual_diagnostics(model_ma.resid, 'MA(q)')
residual_diagnostics(model_ar.resid, 'AR(p)')
residual_diagnostics(model_arma.resid, 'ARMA(p, q)')

# ========================== 13. АНАЛИЗ ТРЕНДА (TS vs DS) НА ОБУЧАЮЩЕЙ ВЫБОРКЕ ==========================
print("\n=== Анализ тренда (TSP vs DSP) ===")
adf_trend = adfuller(train, regression='ct')
print(f"ADF с трендом на train: p-value = {adf_trend[1]:.4f}")
if adf_trend[1] < 0.05:
    print("→ Ряд стационарен вокруг детерминированного тренда (TS).")
else:
    print("→ Ряд имеет стохастический тренд (DS) – требует дифференцирования (что и было сделано).")

def adaptive_forecast(order):
    history = train.copy()
    preds = []

    for i in range(Shag):
        model_step = ARIMA(history, order=order).fit()
        pred = model_step.forecast(steps=1).iloc[0]
        preds.append(pred)
        history = pd.concat([history, pd.Series([test.iloc[i]], index=[test.index[i]])])

    return pd.Series(preds, index=test.index)

# ========================== 14. АДАПТИВНЫЙ ПРОГНОЗ MA(q) ==========================
if used_diff:
    forecast_original = adaptive_forecast((0, 1, best_q))
else:
    forecast_original = adaptive_forecast((0, 0, best_q))
print(f"\n=== Адаптивный прогноз MA(q) на {Shag} шагов ===")
print(forecast_original)

# ========================== 15. АДАПТИВНЫЙ ПРОГНОЗ AR(p) ==========================
if used_diff:
    forecast_ar = adaptive_forecast((best_p, 1, 0))
else:
    forecast_ar = adaptive_forecast((best_p, 0, 0))
print(f"\n=== Адаптивный прогноз AR(p) на {Shag} шагов ===")
print(forecast_ar)

# ========================== 16. АДАПТИВНЫЙ ПРОГНОЗ ARMA(p, q) ==========================
if used_diff:
    forecast_arma = adaptive_forecast((best_p_arma, 1, best_q_arma))
else:
    forecast_arma = adaptive_forecast((best_p_arma, 0, best_q_arma))
print(f"\n=== Адаптивный прогноз ARMA(p, q) на {Shag} шагов ===")
print(forecast_arma)

# ========================== 17. АДАПТИВНЫЕ МЕТОДЫ (на всей исходной обучающей выборке) ==========================
brown_model = ExponentialSmoothing(train, trend=None, seasonal=None).fit()
forecast_brown = brown_model.forecast(Shag)
print("\n=== Метод Брауна (SES) ===")
print(forecast_brown)

holt_model = ExponentialSmoothing(train, trend='add', seasonal=None).fit()
forecast_holt = holt_model.forecast(Shag)
print("=== Метод Хольта ===")
print(forecast_holt)

def calc_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rsse = np.sqrt(np.sum((np.array(y_true) - np.array(y_pred)) ** 2))
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    return mse, rsse, r2, mae
# ============================================================
# ДОПОЛНИТЕЛЬНОЕ РАСШИРЕНИЕ ARIMA-БЛОКА:
# НАИВНАЯ МОДЕЛЬ, ДОВЕРИТЕЛЬНЫЕ ИНТЕРВАЛЫ И GARCH
# ============================================================

print("\n" + "=" * 80)
print("ДОПОЛНИТЕЛЬНЫЙ АНАЛИЗ: BASELINE, ИНТЕРВАЛЫ ARIMA И GARCH")
print("=" * 80)


# ============================================================
# 1. НАИВНАЯ МОДЕЛЬ ПРОГНОЗА
# ============================================================

print("\n" + "=" * 80)
print("НАИВНАЯ МОДЕЛЬ ПРОГНОЗА")
print("=" * 80)

# Наивная модель:
# каждое будущее значение равно последнему значению обучающей выборки.
# Для финансовых рядов это важный baseline, так как цены часто близки
# к случайному блужданию.
naive_forecast = np.repeat(train.iloc[-1], Shag)

naive_mse = mean_squared_error(test, naive_forecast)
naive_rmse = np.sqrt(naive_mse)
naive_mae = mean_absolute_error(test, naive_forecast)
naive_r2 = r2_score(test, naive_forecast)

print("Прогноз наивной модели:")
for i, value in enumerate(naive_forecast, 1):
    print(f"Шаг {i}: {value:.6f}")

print("\nМетрики наивной модели на тестовой выборке:")
print(f"MSE:  {naive_mse:.6f}")
print(f"RMSE: {naive_rmse:.6f}")
print(f"MAE:  {naive_mae:.6f}")
print(f"R2:   {naive_r2:.6f}")

# Визуализация наивного прогноза
plt.figure(figsize=(12, 5))
plt.plot(range(len(train)), train, label="Обучающая выборка", color="gray")
plt.plot(
    range(len(train), len(train) + Shag),
    test,
    "o-",
    label="Факт",
    color="black"
)
plt.plot(
    range(len(train), len(train) + Shag),
    naive_forecast,
    "s--",
    label="Наивный прогноз",
    color="red"
)
plt.title("Наивная модель прогноза")
plt.xlabel("Индекс")
plt.ylabel("Цена акции")
plt.legend()
plt.grid(True, alpha=0.4)
plt.tight_layout()
plt.savefig("naive_forecast.png", dpi=150)
plt.show()


# ============================================================
# 2. ДОВЕРИТЕЛЬНЫЕ ИНТЕРВАЛЫ ДЛЯ ЛУЧШЕЙ ARIMA-МОДЕЛИ
# ============================================================

print("\n" + "=" * 80)
print("ДОВЕРИТЕЛЬНЫЕ ИНТЕРВАЛЫ ПРОГНОЗА ARIMA")
print("=" * 80)

# В твоих расчетах лучшей по метрикам была модель AR(p),
# то есть model_ar = ARIMA(8,1,0), если best_p = 8.
# Поэтому интервалы строим для model_ar.
best_arima_model = model_ar

forecast_res = best_arima_model.get_forecast(steps=Shag)
forecast_mean = forecast_res.predicted_mean
forecast_ci = forecast_res.conf_int(alpha=0.05)

# На случай, если индексы отличаются, приводим всё к numpy
forecast_mean_values = np.asarray(forecast_mean)
lower_ci = np.asarray(forecast_ci.iloc[:, 0])
upper_ci = np.asarray(forecast_ci.iloc[:, 1])

arima_interval_table = pd.DataFrame({
    "Факт": np.asarray(test),
    "Прогноз_ARIMA": forecast_mean_values,
    "Нижняя_граница_95": lower_ci,
    "Верхняя_граница_95": upper_ci
})

print("\nПрогноз ARIMA с 95% доверительными интервалами:")
print(arima_interval_table.round(6).to_string(index=False))

# Проверка попадания фактических значений в доверительный интервал
arima_interval_table["Попадание_в_интервал"] = (
    (arima_interval_table["Факт"] >= arima_interval_table["Нижняя_граница_95"]) &
    (arima_interval_table["Факт"] <= arima_interval_table["Верхняя_граница_95"])
)

coverage = arima_interval_table["Попадание_в_интервал"].mean()

print("\nПроверка попадания фактических значений в 95% доверительный интервал:")
print(arima_interval_table[[
    "Факт",
    "Прогноз_ARIMA",
    "Нижняя_граница_95",
    "Верхняя_граница_95",
    "Попадание_в_интервал"
]].round(6).to_string(index=False))

print(f"\nДоля фактических значений внутри 95% интервала: {coverage:.2%}")

# График ARIMA-прогноза с доверительными интервалами
forecast_index = range(len(train), len(train) + Shag)

plt.figure(figsize=(12, 5))
plt.plot(range(len(train)), train, label="Обучающая выборка", color="gray")
plt.plot(forecast_index, test, "o-", label="Факт", color="black")
plt.plot(forecast_index, forecast_mean_values, "s--", label="Прогноз ARIMA", color="blue")
plt.fill_between(
    forecast_index,
    lower_ci,
    upper_ci,
    alpha=0.25,
    label="95% доверительный интервал"
)
plt.title("Прогноз ARIMA с 95% доверительными интервалами")
plt.xlabel("Индекс")
plt.ylabel("Цена акции")
plt.legend()
plt.grid(True, alpha=0.4)
plt.tight_layout()
plt.savefig("arima_forecast_confidence_interval.png", dpi=150)
plt.show()


# ============================================================
# 3. GARCH(1,1) ДЛЯ АНАЛИЗА ВОЛАТИЛЬНОСТИ
# ============================================================

print("\n" + "=" * 80)
print("МОДЕЛЬ GARCH(1,1) ДЛЯ АНАЛИЗА ВОЛАТИЛЬНОСТИ")
print("=" * 80)

try:
    from arch import arch_model

    # Для GARCH используем первые разности обучающего ряда.
    # Умножаем на 100 для численной устойчивости оценивания.
    garch_series = train.diff().dropna() * 100

    print(f"Длина ряда для GARCH: {len(garch_series)}")
    print(f"Среднее значение ряда первых разностей * 100: {garch_series.mean():.6f}")
    print(f"Стандартное отклонение: {garch_series.std():.6f}")

    # GARCH(1,1) с постоянным средним и нормальным распределением ошибок
    garch_model = arch_model(
        garch_series,
        mean="Constant",
        vol="GARCH",
        p=1,
        q=1,
        dist="normal"
    )

    garch_result = garch_model.fit(disp="off")

    print("\nРезультаты оценки GARCH(1,1):")
    print(garch_result.summary())

    # Извлекаем параметры модели
    garch_params = garch_result.params

    omega = garch_params.get("omega", np.nan)
    alpha_1 = garch_params.get("alpha[1]", np.nan)
    beta_1 = garch_params.get("beta[1]", np.nan)

    persistence = alpha_1 + beta_1

    print("\nКлючевые параметры GARCH(1,1):")
    print(f"omega:       {omega:.6f}")
    print(f"alpha[1]:    {alpha_1:.6f}")
    print(f"beta[1]:     {beta_1:.6f}")
    print(f"alpha+beta:  {persistence:.6f}")

    if persistence < 1:
        print("→ Условие стационарности волатильности выполняется: alpha + beta < 1.")
    else:
        print("→ alpha + beta >= 1: волатильность обладает очень высокой устойчивостью.")

    if persistence > 0.9:
        print("→ Волатильность высокоинерционна: шоки долго влияют на изменчивость ряда.")
    elif persistence > 0.5:
        print("→ Волатильность умеренно инерционна.")
    else:
        print("→ Волатильность быстро возвращается к среднему уровню.")

    # Условная волатильность на обучающей выборке
    conditional_volatility = garch_result.conditional_volatility

    plt.figure(figsize=(12, 5))
    plt.plot(conditional_volatility.index, conditional_volatility.values)
    plt.title("Условная волатильность по модели GARCH(1,1)")
    plt.xlabel("Индекс")
    plt.ylabel("Условная волатильность")
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig("garch_conditional_volatility.png", dpi=150)
    plt.show()

    # Прогноз условной дисперсии и волатильности на Shag шагов
    garch_forecast = garch_result.forecast(horizon=Shag)

    forecast_variance = garch_forecast.variance.iloc[-1].values
    forecast_volatility = np.sqrt(forecast_variance)

    garch_forecast_table = pd.DataFrame({
        "Шаг": np.arange(1, Shag + 1),
        "Прогнозная_дисперсия": forecast_variance,
        "Прогнозная_волатильность": forecast_volatility
    })

    print("\nПрогноз условной волатильности GARCH(1,1) на тестовый период:")
    print(garch_forecast_table.round(6).to_string(index=False))

    # График прогноза волатильности
    plt.figure(figsize=(8, 4))
    plt.plot(
        garch_forecast_table["Шаг"],
        garch_forecast_table["Прогнозная_волатильность"],
        marker="o"
    )
    plt.title("Прогноз условной волатильности GARCH(1,1)")
    plt.xlabel("Шаг прогноза")
    plt.ylabel("Прогнозная волатильность")
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig("garch_volatility_forecast.png", dpi=150)
    plt.show()

    # Стандартизированные остатки GARCH
    standardized_resid = garch_result.std_resid.dropna()

    plt.figure(figsize=(12, 5))
    plt.plot(standardized_resid.index, standardized_resid.values)
    plt.axhline(0, color="black", linestyle="--", linewidth=1)
    plt.title("Стандартизированные остатки GARCH(1,1)")
    plt.xlabel("Индекс")
    plt.ylabel("Стандартизированный остаток")
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig("garch_standardized_residuals.png", dpi=150)
    plt.show()

    # Проверка остатков GARCH на автокорреляцию
    print("\nLjung-Box тест для стандартизированных остатков GARCH:")
    lb_garch_resid = acorr_ljungbox(standardized_resid, lags=[10, 20], return_df=True)
    print(lb_garch_resid)

    print("\nLjung-Box тест для квадратов стандартизированных остатков GARCH:")
    lb_garch_sq_resid = acorr_ljungbox(standardized_resid ** 2, lags=[10, 20], return_df=True)
    print(lb_garch_sq_resid)

except ImportError:
    print(
        "Библиотека arch не установлена. "
        "Установите ее командой: pip install arch"
    )

except Exception as e:
    print(f"При построении GARCH возникла ошибка: {e}")


# ============================================================
# 4. ОБЩАЯ ТАБЛИЦА ДОПОЛНИТЕЛЬНЫХ МЕТРИК
# ============================================================

print("\n" + "=" * 80)
print("ДОПОЛНИТЕЛЬНАЯ СВОДКА")
print("=" * 80)

additional_metrics = pd.DataFrame([
    {
        "Модель": "Наивная модель",
        "MSE": naive_mse,
        "RMSE": naive_rmse,
        "MAE": naive_mae,
        "R2": naive_r2
    }
])

print("\nМетрики baseline-модели:")
print(additional_metrics.round(6).to_string(index=False))

print("\nПримечание:")
print(
    "GARCH-модель не прогнозирует сам уровень цены акции напрямую. "
    "Она используется для моделирования условной волатильности, то есть риска "
    "и изменчивости ряда. Поэтому ее следует рассматривать как дополнение "
    "к ARIMA, а не как прямую замену модели прогноза цены."
)
# ========================== 18. СРАВНЕНИЕ ПРОГНОЗОВ НА ТЕСТОВОЙ ВЫБОРКЕ ==========================
mse_ma, rsse_ma, r2_ma, mae_ma = calc_metrics(test, forecast_original)
mse_ar, rsse_ar, r2_ar, mae_ar = calc_metrics(test, forecast_ar)
mse_arma, rsse_arma, r2_arma, mae_arma = calc_metrics(test, forecast_arma)
mse_naive, rsse_naive, r2_naive, mae_holt = calc_metrics(test, naive_forecast)
mse_brown, rsse_brown, r2_brown, mae_brown = calc_metrics(test, forecast_brown)
mse_holt, rsse_holt, r2_holt, mae_holt = calc_metrics(test, forecast_holt)

print("\n=== Сравнение точности прогнозов на тестовой выборке ===")
print(f"{'Модель':<15} {'MSE':<12} {'RSSE':<12} {'R^2':<12} {'MAE':<12}")
print("-" * 65)
print(f"{'MA(q)':<15} {mse_ma:<12.6f} {rsse_ma:<12.6f} {r2_ma:<12.6f} {mae_ma:<12.6f}")
print(f"{'AR(p)':<15} {mse_ar:<12.6f} {rsse_ar:<12.6f} {r2_ar:<12.6f} {mae_ar:<12.6f}")
print(f"{'ARMA(p,q)':<15} {mse_arma:<12.6f} {rsse_arma:<12.6f} {r2_arma:<12.6f} {mae_arma:<12.6f}")

print(f"{'Naive':<15} {mse_naive:<12.6f} {rsse_naive:<12.6f} {r2_naive:<12.6f} {naive_mae:<12.6f}")
print(f"{'SES':<15} {mse_brown:<12.6f} {rsse_brown:<12.6f} {r2_brown:<12.6f} {mae_brown:<12.6f}")
print(f"{'Holt':<15} {mse_holt:<12.6f} {rsse_holt:<12.6f} {r2_holt:<12.6f} {mae_holt:<12.6f}")


print("\n=== УРАВНЕНИЯ МОДЕЛЕЙ ===")

print("\nMA(q):")
print(build_model_equation(model_ma, used_diff))

print("\nAR(p):")
print(build_model_equation(model_ar, used_diff))

print("\nARMA(p,q):")
print(build_model_equation(model_arma, used_diff))
# ========================== 19. ВИЗУАЛИЗАЦИЯ MA ==========================
test_dates = df['Data'].iloc[-Shag:]
split_date = df['Data'].iloc[-Shag-1]

zoom_points = 30
train_tail = train.iloc[-zoom_points:]
train_dates_tail = df['Data'].iloc[-zoom_points-Shag: -Shag]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [1, 1.5]})

ax1.plot(df['Data'], y, color='gray', linewidth=1, label='Весь ряд')
ax1.scatter(test_dates, test, color='black', s=40, label='Факт (тест)')
ax1.plot(test_dates, forecast_original, 's--', color='blue', label='MA(q)')
ax1.axvline(x=split_date, color='black', linestyle=':', linewidth=1)
ax1.set_title(f'Полный ряд с прогнозом MA(q) на {Shag} шагов')
ax1.legend(loc='upper left')
ax1.grid(True, alpha=0.3)

ax2.plot(train_dates_tail, train_tail, color='gray', linewidth=1.5, label='Обучающая выборка (хвост)')
ax2.scatter(test_dates, test, color='black', s=80, zorder=5, label='Факт')
ax2.plot(test_dates, forecast_original, '--', color='blue', linewidth=2, markersize=8, label='MA(q)')
ax2.axvline(x=split_date, color='black', linestyle=':', linewidth=1.5)
ax2.set_title(f'Увеличенный фрагмент MA(q) (последние {zoom_points} наблюдений + прогноз)')
ax2.grid(True, alpha=0.5)
ax2.set_ylabel('Цена акции')
ax2.legend(loc='best')
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig('forecast_ma_two_panels.png', dpi=200)
plt.show()

# ========================== 20. ВИЗУАЛИЗАЦИЯ AR ==========================
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [1, 1.5]})

ax1.plot(df['Data'], y, color='gray', linewidth=1, label='Весь ряд')
ax1.scatter(test_dates, test, color='black', s=40, label='Факт (тест)')
ax1.plot(test_dates, forecast_ar, 's--', color='darkorange', label='AR(p)')
ax1.axvline(x=split_date, color='black', linestyle=':', linewidth=1)
ax1.set_title(f'Полный ряд с прогнозом AR(p) на {Shag} шагов')
ax1.legend(loc='upper left')
ax1.grid(True, alpha=0.3)

ax2.plot(train_dates_tail, train_tail, color='gray', linewidth=1.5, label='Обучающая выборка (хвост)')
ax2.scatter(test_dates, test, color='black', s=80, zorder=5, label='Факт')
ax2.plot(test_dates, forecast_ar, '--', color='darkorange', linewidth=2, markersize=8, label='AR(p)')
ax2.axvline(x=split_date, color='black', linestyle=':', linewidth=1.5)
ax2.set_title(f'Увеличенный фрагмент AR(p) (последние {zoom_points} наблюдений + прогноз)')
ax2.grid(True, alpha=0.5)
ax2.set_ylabel('Цена акции')
ax2.legend(loc='best')
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig('forecast_ar_two_panels.png', dpi=200)
plt.show()

# ========================== 21. ВИЗУАЛИЗАЦИЯ ARMA ==========================
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [1, 1.5]})

ax1.plot(df['Data'], y, color='gray', linewidth=1, label='Весь ряд')
ax1.scatter(test_dates, test, color='black', s=40, label='Факт (тест)')
ax1.plot(test_dates, forecast_arma, 's--', color='purple', label='ARMA(p,q)')
ax1.axvline(x=split_date, color='black', linestyle=':', linewidth=1)
ax1.set_title(f'Полный ряд с прогнозом ARMA(p,q) на {Shag} шагов')
ax1.legend(loc='upper left')
ax1.grid(True, alpha=0.3)

ax2.plot(train_dates_tail, train_tail, color='gray', linewidth=1.5, label='Обучающая выборка (хвост)')
ax2.scatter(test_dates, test, color='black', s=80, zorder=5, label='Факт')
ax2.plot(test_dates, forecast_arma, '--', color='purple', linewidth=2, markersize=8, label='ARMA(p,q)')
ax2.axvline(x=split_date, color='black', linestyle=':', linewidth=1.5)
ax2.set_title(f'Увеличенный фрагмент ARMA(p,q) (последние {zoom_points} наблюдений + прогноз)')
ax2.grid(True, alpha=0.5)
ax2.set_ylabel('Цена акции')
ax2.legend(loc='best')
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig('forecast_arma_two_panels.png', dpi=200)
plt.show()

# ========================== 22. ОБЩИЙ РИСУНОК ДЛЯ ВСЕХ МОДЕЛЕЙ ==========================
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [1, 1.5]})

ax1.plot(df['Data'], y, color='gray', linewidth=1, label='Весь ряд')
ax1.scatter(test_dates, test, color='black', s=40, label='Факт (тест)')
ax1.plot(test_dates, forecast_original, 's--', color='blue', label='MA(q)')
ax1.plot(test_dates, forecast_ar, 'o--', color='darkorange', label='AR(p)')
ax1.plot(test_dates, forecast_arma, 'd--', color='purple', label='ARMA(p,q)')
ax1.plot(test_dates, forecast_brown, '^--', color='green', label='SES')
ax1.plot(test_dates, forecast_holt, 'x--', color='red', label='Хольт')
ax1.axvline(x=split_date, color='black', linestyle=':', linewidth=1)
ax1.set_title(f'Полный ряд с прогнозами всех моделей на {Shag} шагов')
ax1.legend(loc='upper left')
ax1.grid(True, alpha=0.3)

ax2.plot(train_dates_tail, train_tail, color='gray', linewidth=1.5, label='Обучающая выборка (хвост)')
ax2.scatter(test_dates, test, color='black', s=80, zorder=5, label='Факт')
ax2.plot(test_dates, forecast_original, 's--', color='blue', linewidth=2, markersize=8, label='MA(q)')
ax2.plot(test_dates, forecast_ar, 'o--', color='darkorange', linewidth=2, markersize=8, label='AR(p)')
ax2.plot(test_dates, forecast_arma, 'd--', color='purple', linewidth=2, markersize=8, label='ARMA(p,q)')
ax2.plot(test_dates, forecast_brown, '^--', color='green', linewidth=2, markersize=8, label='SES')
ax2.plot(test_dates, forecast_holt, 'x--', color='red', linewidth=2, markersize=8, label='Хольт')
ax2.axvline(x=split_date, color='black', linestyle=':', linewidth=1.5)
ax2.set_title(f'Увеличенный фрагмент: сравнение всех моделей (последние {zoom_points} наблюдений + прогноз)')
ax2.grid(True, alpha=0.5)
ax2.set_ylabel('Цена акции')
ax2.legend(loc='best')
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig('forecast_all_models_two_panels.png', dpi=200)
plt.show()