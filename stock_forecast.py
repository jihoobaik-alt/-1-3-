from datetime import datetime, timedelta, timezone
from io import BytesIO
from urllib.request import urlopen
from zipfile import ZipFile

import exchange_calendars as xcals
import numpy as np
import pandas as pd
import yfinance as yf


# ============================================================
# 설정
# ============================================================
START = "2023-01-01"
END = "2027-01-01"  # 종료일 제외. 계속 최신 자료를 쓰려면 None

KST = timezone(timedelta(hours=9))

COLUMNS = ["Open", "High", "Low", "Close", "Volume"]
PRICE_COLUMNS = COLUMNS[:4]

KOREAN_NAMES = {
    "Open": "시가",
    "High": "고가",
    "Low": "저가",
    "Close": "종가",
    "Volume": "거래량",
}

N_TREES = 32
MAX_DEPTH = 6
MIN_SAMPLES_LEAF = 8
MAX_THRESHOLDS = 16

MIN_DATA_ROWS = 250
TEST_RATIO = 0.20
RANDOM_SEED = 42


def normalize(text):
    return "".join(str(text).split()).casefold()


def today_kst():
    return pd.Timestamp(datetime.now(KST).date())


def request_period():
    """실행 당일과 미래 자료를 제외한 요청 범위."""
    today = today_kst()
    end = today if END is None else min(pd.Timestamp(END), today)

    if pd.Timestamp(START) >= end:
        raise ValueError("조회 시작일은 종료일보다 이전이어야 합니다.")

    return START, end.strftime("%Y-%m-%d")


# ============================================================
# 1. 기업명 검색
# ============================================================
def download_stock_list():
    stocks = {}

    for market, suffix in [("kospi", "KS"), ("kosdaq", "KQ")]:
        url = (
            "https://new.real.download.dws.co.kr/common/master/"
            f"{market}_code.mst.zip"
        )

        with urlopen(url, timeout=30) as response:
            with ZipFile(BytesIO(response.read())) as archive:
                data = archive.read(f"{market}_code.mst")

        for row in data.splitlines():
            if len(row) < 61:
                continue

            try:
                code = row[:9].decode("ascii").strip()[-6:]
                name = row[21:61].decode("cp949").strip()
            except UnicodeDecodeError:
                continue

            if len(code) == 6 and code.isalnum() and name:
                stocks[f"{code}.{suffix}"] = name

    if not stocks:
        raise RuntimeError("종목 목록을 읽지 못했습니다.")

    return sorted(
        [(name, symbol) for symbol, name in stocks.items()],
        key=lambda item: normalize(item[0]),
    )


def search_stocks(stocks, query):
    query = normalize(query)

    query = {
        "네이버": "naver",
        "현대자동차": "현대차",
    }.get(query, query)

    if not query:
        raise ValueError("기업 이름을 입력해 주세요.")

    # 기업명, 전체 티커, 6자리 종목코드의 정확한 일치 우선
    exact = [
        (name, symbol)
        for name, symbol in stocks
        if query in {
            normalize(name),
            normalize(symbol),
            normalize(symbol.split(".")[0]),
        }
    ]

    matches = exact or [
        (name, symbol)
        for name, symbol in stocks
        if query in normalize(name) or query in normalize(symbol)
    ]

    if not matches:
        raise ValueError(
            "검색 결과가 없어요. "
            "국내 코스피·코스닥 기업명이나 종목코드를 확인해 주세요."
        )

    return matches


# ============================================================
# 2. 과거 주가 수집
#    서버의 공용 폴더에 저장하지 않고 자료를 반환합니다.
# ============================================================
def load_prices(symbol, start, end):
    raw = yf.Ticker(symbol).history(
        start=start,
        end=end,
        interval="1d",
        auto_adjust=False,
        actions=False,
        repair=False,
        keepna=True,
        timeout=30,
    )

    if raw is None or raw.empty:
        raise RuntimeError(
            "주가 자료를 받지 못했어요. "
            "연결 상태, 종목 지원 여부 또는 데이터 제공처의 "
            "요청 제한을 확인하고 잠시 후 다시 시도해 주세요."
        )

    if not set(COLUMNS).issubset(raw.columns):
        raise RuntimeError("시가·고가·저가·종가·거래량 열이 필요합니다.")

    df = raw[COLUMNS].copy()

    index = pd.DatetimeIndex(df.index)

    if index.tz is not None:
        index = index.tz_convert(KST).tz_localize(None)

    df.index = index.normalize()
    df.index.name = "Date"

    df = df.loc[
        ~df.index.duplicated(keep="last")
    ].sort_index()

    end_date = pd.Timestamp(end)

    df = df.loc[
        (df.index >= pd.Timestamp(start))
        & (df.index < end_date)
    ].copy()

    if df.empty:
        raise RuntimeError("설정한 기간에 수집된 자료가 없습니다.")

    df = df.apply(pd.to_numeric, errors="coerce").astype(float)
    df = df.replace([np.inf, -np.inf], np.nan)

    # 한국거래소 달력으로 중간에 누락된 거래일도 확인합니다.
    calendar_end = end_date + pd.Timedelta(days=40)

    try:
        calendar = xcals.get_calendar(
            "XKRX",
            start=df.index[0],
            end=calendar_end,
        )
        sessions = pd.DatetimeIndex(calendar.sessions)
        sessions = sessions.tz_localize(None).normalize()

        sessions = sessions[
            (sessions >= df.index[0])
            & (sessions <= calendar_end)
        ]
    except Exception as exc:
        raise RuntimeError(
            "한국거래소 거래일 달력을 읽지 못했습니다. "
            "exchange-calendars 패키지와 날짜 범위를 확인해 주세요."
        ) from exc

    if len(df.index.difference(sessions)) > 0:
        raise RuntimeError(
            "수집 날짜와 거래소 달력이 일치하지 않습니다. "
            "달력 패키지를 업데이트하고 원자료를 확인해 주세요."
        )

    past_sessions = sessions[sessions < end_date]

    if len(past_sessions) == 0:
        raise RuntimeError("사용할 수 있는 과거 거래일이 없습니다.")

    # 누락 행을 삭제해서 날짜가 당겨지지 않도록 유지합니다.
    df = df.reindex(past_sessions)
    df.index.name = "Date"

    invalid = (
        df.isna().any(axis=1)
        | df[PRICE_COLUMNS].le(0).any(axis=1)
        | df["Volume"].le(0)
        | df["High"].lt(df["Low"])
        | df["High"].lt(df[["Open", "Close"]].max(axis=1))
        | df["Low"].gt(df[["Open", "Close"]].min(axis=1))
    )

    issues = df.loc[invalid].copy()

    # 이 예제에서는 거래량 0인 날도 모델에서 제외합니다.
    # 결측·거래정지 등을 임의 보간하지 않습니다.
    df.loc[invalid, COLUMNS] = np.nan

    if df.dropna(subset=COLUMNS).empty:
        raise RuntimeError("사용 가능한 OHLCV 자료가 없습니다.")

    if df.iloc[-1].isna().any():
        raise RuntimeError(
            f"최근 거래일 {df.index[-1]:%Y-%m-%d}의 "
            "자료가 없거나 거래량이 0입니다. "
            "공급 지연·거래정지·결측 여부를 확인해 주세요. "
            "오래된 자료로 대신 예측하지 않습니다."
        )

    next_sessions = sessions[sessions > df.index[-1]]

    if len(next_sessions) == 0:
        raise RuntimeError("다음 거래일을 찾지 못했습니다.")

    return {
        "raw": raw,
        "grid": df,
        "issues": issues,
        "target_date": next_sessions[0],
        "fetched_at": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S"),
    }


# ============================================================
# 3. 회귀용 의사결정트리
# ============================================================
def checked_xy(X, y):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    if y.ndim == 1:
        y = y.reshape(-1, 1)

    if (
        X.ndim != 2
        or y.ndim != 2
        or X.size == 0
        or y.size == 0
        or len(X) != len(y)
    ):
        raise ValueError("학습 배열의 크기를 확인해 주세요.")

    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError("학습 자료에 결측값 또는 무한대가 있습니다.")

    return X, y


class Node:
    def __init__(
        self,
        feature=None,
        threshold=None,
        left=None,
        right=None,
        *,
        value=None,
    ):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        return self.value is not None


class DecisionTreeRegressor:
    def __init__(
        self,
        max_depth=6,
        min_samples_leaf=8,
        n_features=None,
        max_thresholds=16,
        random_state=None,
    ):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.n_features = n_features
        self.max_thresholds = max_thresholds
        self.random_state = random_state
        self.root = None

    def fit(self, X, y):
        X, y = checked_xy(X, y)
        self.rng = np.random.default_rng(self.random_state)
        self.n_features_in_ = X.shape[1]

        self.n_features_ = (
            X.shape[1]
            if self.n_features is None
            else max(1, min(int(self.n_features), X.shape[1]))
        )

        self.root = self._grow_tree(X, y)
        return self

    @staticmethod
    def _mse(y):
        return float(np.var(y, axis=0).mean())

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        leaf = Node(value=y.mean(axis=0))
        parent_loss = self._mse(y)

        if (
            depth >= self.max_depth
            or n_samples < 2 * self.min_samples_leaf
            or parent_loss <= 1e-12
        ):
            return leaf

        feature_indices = self.rng.choice(
            n_features,
            self.n_features_,
            replace=False,
        )

        best_loss = parent_loss
        best_split = None

        for feature in feature_indices:
            values = np.unique(X[:, feature])

            if len(values) < 2:
                continue

            thresholds = values[:-1] + np.diff(values) / 2

            # 실행 시간을 줄이기 위한 근사 분할 탐색
            if len(thresholds) > self.max_thresholds:
                positions = np.linspace(
                    0,
                    len(thresholds) - 1,
                    self.max_thresholds,
                    dtype=int,
                )
                thresholds = thresholds[positions]

            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                n_left = int(left_mask.sum())
                n_right = n_samples - n_left

                if (
                    n_left < self.min_samples_leaf
                    or n_right < self.min_samples_leaf
                ):
                    continue

                loss = (
                    n_left * self._mse(y[left_mask])
                    + n_right * self._mse(y[~left_mask])
                ) / n_samples

                if loss < best_loss - 1e-12:
                    best_loss = loss
                    best_split = (
                        feature,
                        float(threshold),
                        left_mask,
                    )

        if best_split is None:
            return leaf

        feature, threshold, left_mask = best_split

        return Node(
            feature=feature,
            threshold=threshold,
            left=self._grow_tree(
                X[left_mask], y[left_mask], depth + 1
            ),
            right=self._grow_tree(
                X[~left_mask], y[~left_mask], depth + 1
            ),
        )

    def predict(self, X):
        if self.root is None:
            raise RuntimeError("먼저 모델을 학습해야 합니다.")

        X = np.asarray(X, dtype=float)

        if (
            X.ndim != 2
            or X.shape[1] != self.n_features_in_
            or not np.isfinite(X).all()
        ):
            raise ValueError("예측 입력을 확인해 주세요.")

        predictions = []

        for x in X:
            node = self.root

            while not node.is_leaf_node():
                node = (
                    node.left
                    if x[node.feature] <= node.threshold
                    else node.right
                )

            predictions.append(node.value)

        return np.asarray(predictions)


# ============================================================
# 4. 회귀용 랜덤 포레스트
# ============================================================
class RandomForestRegressor:
    def __init__(
        self,
        n_trees=N_TREES,
        max_depth=MAX_DEPTH,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        max_thresholds=MAX_THRESHOLDS,
        random_state=RANDOM_SEED,
    ):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_thresholds = max_thresholds
        self.random_state = random_state
        self.trees = []

    def fit(self, X, y, on_tree=None):
        X, y = checked_xy(X, y)

        if self.n_trees < 1:
            raise ValueError("트리는 1개 이상이어야 합니다.")

        self.trees = []
        rng = np.random.default_rng(self.random_state)

        # 거래량이 가격 변화율보다 숫자가 훨씬 크기 때문에
        # 출력 열을 학습 자료 기준으로 표준화합니다.
        self.y_mean_ = y.mean(axis=0)
        scale = y.std(axis=0)
        self.y_scale_ = np.where(scale > 1e-12, scale, 1.0)

        standardized_y = (y - self.y_mean_) / self.y_scale_

        n_samples, n_features = X.shape
        selected_features = max(1, int(np.sqrt(n_features)))

        for number in range(self.n_trees):
            indices = rng.choice(
                n_samples,
                size=n_samples,
                replace=True,
            )

            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                n_features=selected_features,
                max_thresholds=self.max_thresholds,
                random_state=int(rng.integers(0, 2**32 - 1)),
            )

            tree.fit(X[indices], standardized_y[indices])
            self.trees.append(tree)

            if on_tree is not None:
                on_tree(number + 1, self.n_trees)

        return self

    def predict(self, X):
        if not self.trees:
            raise RuntimeError("먼저 모델을 학습해야 합니다.")

        prediction = np.mean(
            [tree.predict(X) for tree in self.trees],
            axis=0,
        )

        return prediction * self.y_scale_ + self.y_mean_


# ============================================================
# 5. 입력 특성 생성
# ============================================================
def make_features(df):
    close = df["Close"]
    volume = df["Volume"]
    daily_return = close.pct_change(fill_method=None)

    X = pd.DataFrame(index=df.index)

    for column in ["Open", "High", "Low"]:
        X[f"{column}_relative_close"] = df[column] / close - 1

    X["daily_range"] = (df["High"] - df["Low"]) / close

    for period in [1, 2, 3, 5, 10]:
        X[f"return_{period}"] = close.pct_change(
            periods=period,
            fill_method=None,
        )

    for lag in range(1, 6):
        X[f"return_lag_{lag}"] = daily_return.shift(lag)

    for window in [5, 10, 20, 60]:
        X[f"close_to_ma_{window}"] = (
            close / close.rolling(window).mean() - 1
        )

    for window in [5, 20]:
        X[f"volatility_{window}"] = (
            daily_return.rolling(window).std()
        )

    X["log_volume"] = np.log1p(volume)
    X["log_volume_change"] = np.log1p(volume).diff()

    for window in [5, 20]:
        X[f"volume_relative_{window}"] = (
            volume / volume.rolling(window).mean() - 1
        )

    X["weekday"] = X.index.dayofweek

    return X.replace([np.inf, -np.inf], np.nan)


def restore_predictions(predictions, base_close):
    result = np.asarray(predictions, dtype=float).copy()
    base_close = np.asarray(
        base_close, dtype=float
    ).reshape(-1, 1)

    # 시가·고가·저가·종가: 기준일 종가 대비 변화율에서 복원
    result[:, :4] = base_close * (1 + result[:, :4])
    result[:, :4] = np.maximum(result[:, :4], 1e-8)

    # OHLC 관계를 형식적으로 보정합니다.
    result[:, 1] = np.maximum(
        result[:, 1],
        np.maximum(result[:, 0], result[:, 3]),
    )
    result[:, 2] = np.minimum(
        result[:, 2],
        np.minimum(result[:, 0], result[:, 3]),
    )

    result[:, 4] = np.maximum(0, np.rint(result[:, 4]))

    return result


# ============================================================
# 6. 검증 및 최종 예측
# ============================================================
def train_and_forecast(bundle, name, symbol, report=None):
    def notify(percent, text):
        if report is not None:
            report(int(percent), text)

    df = bundle["grid"]

    notify(10, "입력 특성과 다음 거래일 정답을 만들고 있어요.")

    features = make_features(df)
    next_values = df[COLUMNS].shift(-1)

    targets = next_values.copy()
    targets[PRICE_COLUMNS] = (
        next_values[PRICE_COLUMNS].div(df["Close"], axis=0) - 1
    )
    targets.columns = [
        "Open_Return",
        "High_Return",
        "Low_Return",
        "Close_Return",
        "Volume",
    ]
    targets = targets.replace([np.inf, -np.inf], np.nan)

    target_dates = pd.Series(
        df.index,
        index=df.index,
        name="TargetDate",
    ).shift(-1)

    valid = (
        features.notna().all(axis=1)
        & targets.notna().all(axis=1)
    )

    if int(valid.sum()) < MIN_DATA_ROWS:
        raise RuntimeError(
            f"유효한 학습 사례가 {MIN_DATA_ROWS}개 미만입니다. "
            "자료가 충분한 다른 종목을 선택하거나 "
            "START 날짜를 앞당겨 주세요."
        )

    if features.iloc[-1].isna().any():
        raise RuntimeError(
            "최신 날짜의 입력 특성을 만들 수 없습니다. "
            "최근 60거래일의 결측 자료를 확인해 주세요."
        )

    X = features.loc[valid].to_numpy(dtype=float)
    y = targets.loc[valid].to_numpy(dtype=float)

    sample_dates = features.index[valid]
    base_close = df.loc[valid, "Close"].to_numpy(dtype=float)

    actual_next = next_values.loc[valid].to_numpy(dtype=float)
    baseline = df.loc[valid, COLUMNS].to_numpy(dtype=float)

    learning_data = features.loc[valid].join(
        targets.loc[valid].add_prefix("Target_")
    )
    learning_data.insert(
        0, "TargetDate", target_dates.loc[valid]
    )
    learning_data = learning_data.rename_axis("InputDate")

    # 날짜 순서를 유지한 검증
    split = int(len(X) * (1 - TEST_RATIO))
    first_test_date = sample_dates[split]

    # 학습 정답 날짜도 첫 검증 입력일보다 이전으로 제한
    train_mask = (
        target_dates.loc[valid] < first_test_date
    ).to_numpy()

    if int(train_mask.sum()) < 2:
        raise RuntimeError("학습 구간이 너무 짧습니다.")

    forest = RandomForestRegressor()

    forest.fit(
        X[train_mask],
        y[train_mask],
        on_tree=lambda done, total: notify(
            15 + 30 * done // total,
            f"검증용 모델 학습 중 · 트리 {done}/{total}",
        ),
    )

    test_prediction = restore_predictions(
        forest.predict(X[split:]),
        base_close[split:],
    )

    actual_test = actual_next[split:]
    baseline_test = baseline[split:]

    model_mae = np.abs(
        test_prediction - actual_test
    ).mean(axis=0)

    baseline_mae = np.abs(
        baseline_test - actual_test
    ).mean(axis=0)

    denominator = np.where(
        baseline_mae > 0,
        baseline_mae,
        np.nan,
    )

    metrics = pd.DataFrame(
        {
            "RF_MAE": model_mae,
            "전일값_MAE": baseline_mae,
            "개선율(%)": 100 * (1 - model_mae / denominator),
        },
        index=[KOREAN_NAMES[c] for c in COLUMNS],
    ).rename_axis("항목")

    test_details = pd.DataFrame(
        index=pd.DatetimeIndex(
            target_dates.loc[valid].iloc[split:].to_numpy(),
            name="Date",
        )
    )
    test_details["InputDate"] = sample_dates[split:].to_numpy()

    for j, column in enumerate(COLUMNS):
        test_details[f"Actual_{column}"] = actual_test[:, j]
        test_details[f"Predicted_{column}"] = test_prediction[:, j]
        test_details[f"Baseline_{column}"] = baseline_test[:, j]

    # 검증 후 정답을 아는 전체 사례로 재학습
    forest.fit(
        X,
        y,
        on_tree=lambda done, total: notify(
            55 + 40 * done // total,
            f"최종 모델 학습 중 · 트리 {done}/{total}",
        ),
    )

    last_close = float(df["Close"].iloc[-1])
    latest_X = features.iloc[[-1]].to_numpy(dtype=float)

    prediction = restore_predictions(
        forest.predict(latest_X),
        [last_close],
    )[0]

    change_pct = float(
        (prediction[3] / last_close - 1) * 100
    )

    forecast = pd.DataFrame(
        [prediction],
        columns=COLUMNS,
        index=pd.DatetimeIndex(
            [bundle["target_date"]],
            name="Date",
        ),
    )
    forecast.insert(0, "InputDate", df.index[-1])
    forecast.insert(0, "Symbol", symbol)
    forecast.insert(0, "Name", name)
    forecast["Close_Change_Pct"] = change_pct

    notify(100, "예측을 완료했어요.")

    return {
        "name": name,
        "symbol": symbol,
        "input_date": df.index[-1],
        "target_date": bundle["target_date"],
        "last_close": last_close,
        "prediction": prediction,
        "change_pct": change_pct,
        "metrics": metrics,
        "forecast": forecast,
        "prices": df.dropna(subset=COLUMNS),
        "raw": bundle["raw"],
        "issues": bundle["issues"],
        "learning_data": learning_data,
        "test_predictions": test_details,
        "n_samples": len(X),
        "n_train": int(train_mask.sum()),
        "n_test": len(X) - split,
        "fetched_at": bundle["fetched_at"],
    }
