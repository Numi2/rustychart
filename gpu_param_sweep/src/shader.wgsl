struct Ohlc {
    open: f32;
    high: f32;
    low: f32;
    close: f32;
};

struct OhlcBuf {
    data: array<Ohlc>;
};

struct ParamGrid {
    // NOTE: layout is mirrored exactly in Rust (ParamGridUniform).
    emaMin: f32;
    emaStep: f32;
    emaCount: u32;

    bandMin: f32;
    bandStep: f32;
    bandCount: u32;

    stopMin: f32;
    stopStep: f32;
    stopCount: u32;

    targetMin: f32;
    targetStep: f32;
    targetCount: u32;

    riskMin: f32;
    riskStep: f32;
    riskCount: u32;

    numBars: u32;
    numCombos: u32;
    costPerTrade: f32;
    slippageBps: f32;
    pad0: u32;
    pad1: u32;
};

struct Result {
    finalEquity: f32;
    profitFactor: f32;
    sharpe: f32;
    maxDrawdown: f32;
    numTrades: u32;
    winTrades: u32;
    lossTrades: u32;
    totalTradeBars: u32;
    barsWithPosition: u32;
    _pad: u32;
    _pad2: u32;
};

struct ResultsBuf {
    data: array<Result>;
};

@group(0) @binding(0)
var<storage, read> ohlc: OhlcBuf;

@group(0) @binding(1)
var<uniform> grid: ParamGrid;

@group(0) @binding(2)
var<storage, read_write> results: ResultsBuf;

fn clampPositive(x: f32, minVal: f32) -> f32 {
    if (x < minVal) {
        return minVal;
    }
    return x;
}

override WORKGROUP_SIZE: u32 = 256u;

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let comboIndex: u32 = gid.x;
    let numCombos: u32 = grid.numCombos;

    if (comboIndex >= numCombos) {
        return;
    }

    let emaCount: u32    = grid.emaCount;
    let bandCount: u32   = grid.bandCount;
    let stopCount: u32   = grid.stopCount;
    let targetCount: u32 = grid.targetCount;
    let riskCount: u32   = grid.riskCount;
    let numBars: u32     = grid.numBars;

    if (numBars < 2u) {
        results.data[comboIndex] = Result(
            finalEquity: 1.0,
            profitFactor: 0.0,
            sharpe: 0.0,
            maxDrawdown: 0.0,
            numTrades: 0u,
            winTrades: 0u,
            lossTrades: 0u,
            totalTradeBars: 0u,
            barsWithPosition: 0u,
            _pad: 0u,
            _pad2: 0u
        );
        return;
    }

    // Decode 1D index -> 5D grid indices
    var idx: u32 = comboIndex;

    let emaIdx: u32 = idx % emaCount;
    idx = idx / emaCount;

    let bandIdx: u32 = idx % bandCount;
    idx = idx / bandCount;

    let stopIdx: u32 = idx % stopCount;
    idx = idx / stopCount;

    let targetIdx: u32 = idx % targetCount;
    idx = idx / targetCount;

    let riskIdx: u32 = idx % riskCount;

    // Parameter values from ranges
    let emaLen: f32       = clampPositive(grid.emaMin     + f32(emaIdx)    * grid.emaStep,     1.0);
    let bandWidth: f32    = clampPositive(grid.bandMin    + f32(bandIdx)   * grid.bandStep,    0.0);
    let stopMult: f32     = clampPositive(grid.stopMin    + f32(stopIdx)   * grid.stopStep,    0.01);
    let targetMult: f32   = clampPositive(grid.targetMin  + f32(targetIdx) * grid.targetStep,  0.01);
    let riskPerTrade: f32 = clampPositive(grid.riskMin    + f32(riskIdx)   * grid.riskStep,    0.00001);

    // Indicators
    let alphaEma: f32 = 2.0 / (emaLen + 1.0);
    let atrLen: f32   = emaLen;
    let alphaAtr: f32 = 1.0 / atrLen; // Wilder-style RMA

    var firstBar: Ohlc = ohlc.data[0u];
    var ema: f32       = firstBar.close;
    var prevClose: f32 = ema;
    var tr0: f32       = abs(firstBar.high - firstBar.low);
    var atr: f32       = tr0;

    // Equity / trade state
    var equity: f32     = 1.0;
    var equityPeak: f32 = 1.0;
    var maxDD: f32      = 0.0;

    var posDirection: f32 = 0.0; // +1 long, -1 short, 0 flat
    var entryPrice: f32   = 0.0;
    var stopPrice: f32    = 0.0;
    var targetPrice: f32  = 0.0;
    var entryEquity: f32  = 0.0;

    var trades: u32      = 0u;
    var winTrades: u32   = 0u;
    var lossTrades: u32  = 0u;
    var grossProfit: f32 = 0.0;
    var grossLoss: f32   = 0.0;

    // Welford stats for trade R
    var meanR: f32 = 0.0;
    var m2R: f32   = 0.0;
    var totalTradeBars: u32 = 0u;
    var currentTradeBars: u32 = 0u;
    var barsWithPosition: u32 = 0u;

    // Main loop
    for (var i: u32 = 1u; i < numBars; i = i + 1u) {
        let bar: Ohlc   = ohlc.data[i];
        let close: f32  = bar.close;
        let high: f32   = bar.high;
        let low: f32    = bar.low;

        if (posDirection != 0.0) {
            currentTradeBars = currentTradeBars + 1u;
            barsWithPosition = barsWithPosition + 1u;
        }

        // EMA update
        ema = alphaEma * close + (1.0 - alphaEma) * ema;

        // ATR update
        let highLow: f32  = high - low;
        let highPrev: f32 = abs(high - prevClose);
        let lowPrev: f32  = abs(low - prevClose);
        let tr1: f32      = max(highLow, max(highPrev, lowPrev));
        atr = (atr * (atrLen - 1.0) + tr1) * alphaAtr;
        prevClose = close;

        let upperBand: f32 = ema + bandWidth * atr;
        let lowerBand: f32 = ema - bandWidth * atr;

        // Manage open position
        if (posDirection != 0.0) {
            var exit: bool      = false;
            var exitPrice: f32  = close;

            if (posDirection > 0.0) {
                if (high >= targetPrice) {
                    exit = true;
                    exitPrice = targetPrice;
                    winTrades = winTrades + 1u;
                } else if (low <= stopPrice) {
                    exit = true;
                    exitPrice = stopPrice;
                    lossTrades = lossTrades + 1u;
                }
            } else {
                if (low <= targetPrice) {
                    exit = true;
                    exitPrice = targetPrice;
                    winTrades = winTrades + 1u;
                } else if (high >= stopPrice) {
                    exit = true;
                    exitPrice = stopPrice;
                    lossTrades = lossTrades + 1u;
                }
            }

            if (exit) {
                let stopDist: f32 = abs(entryPrice - stopPrice);
                let size: f32     = (entryEquity * riskPerTrade) / max(stopDist, 1e-6);
                let pnl: f32      = (exitPrice - entryPrice) * posDirection * size;
                let notional: f32 = abs(entryPrice) * size;
                let slippageCost: f32 = notional * grid.slippageBps * 0.0001 * 2.0;
                let pnlNet: f32   = pnl - grid.costPerTrade - slippageCost;
                equity             = equity + pnlNet;

                trades = trades + 1u;

                let eqBefore: f32 = entryEquity;
                let r: f32        = pnlNet / max(eqBefore, 1e-6);
                let t: f32        = f32(trades);
                let delta: f32    = r - meanR;
                meanR             = meanR + delta / t;
                let delta2: f32   = r - meanR;
                m2R               = m2R + delta * delta2;

                if (pnlNet > 0.0) {
                    grossProfit = grossProfit + pnlNet;
                } else {
                    grossLoss = grossLoss - pnlNet;
                }

                if (equity > equityPeak) {
                    equityPeak = equity;
                } else {
                    let dd: f32 = (equityPeak - equity) / max(equityPeak, 1e-6);
                    if (dd > maxDD) {
                        maxDD = dd;
                    }
                }

                totalTradeBars = totalTradeBars + currentTradeBars;
                currentTradeBars = 0u;
                posDirection = 0.0;
            }
        }

        // Entries (only if flat)
        if (posDirection == 0.0) {
            if (close > upperBand) {
                posDirection = 1.0;
                entryPrice   = close;
                entryEquity  = equity;
                stopPrice    = close - stopMult * atr;
                targetPrice  = close + targetMult * atr;
            } else if (close < lowerBand) {
                posDirection = -1.0;
                entryPrice   = close;
                entryEquity  = equity;
                stopPrice    = close + stopMult * atr;
                targetPrice  = close - targetMult * atr;
            }
        }
    }

    // Exit any open position at final close
    if (posDirection != 0.0) {
        let lastBar: Ohlc = ohlc.data[numBars - 1u];
        let close: f32    = lastBar.close;

        let stopDist: f32 = abs(entryPrice - stopPrice);
        let size: f32     = (entryEquity * riskPerTrade) / max(stopDist, 1e-6);
        let pnl: f32      = (close - entryPrice) * posDirection * size;
        let notional: f32 = abs(entryPrice) * size;
        let slippageCost: f32 = notional * grid.slippageBps * 0.0001 * 2.0;
        let pnlNet: f32   = pnl - grid.costPerTrade - slippageCost;
        equity             = equity + pnlNet;

        trades = trades + 1u;

        let eqBefore: f32 = entryEquity;
        let r: f32        = pnlNet / max(eqBefore, 1e-6);
        let t: f32        = f32(trades);
        let delta: f32    = r - meanR;
        meanR             = meanR + delta / t;
        let delta2: f32   = r - meanR;
        m2R               = m2R + delta * delta2;

        if (pnlNet > 0.0) {
            grossProfit = grossProfit + pnlNet;
        } else {
            grossLoss = grossLoss - pnlNet;
        }

        if (equity > equityPeak) {
            equityPeak = equity;
        } else {
            let dd: f32 = (equityPeak - equity) / max(equityPeak, 1e-6);
            if (dd > maxDD) {
                maxDD = dd;
            }
        }
        totalTradeBars = totalTradeBars + currentTradeBars;
        currentTradeBars = 0u;
    }

    // Metrics
    var pf: f32 = 0.0;
    if (grossLoss > 0.0) {
        pf = grossProfit / grossLoss;
    }

    var sharpe: f32 = 0.0;
    if (trades > 1u) {
        let variance: f32 = m2R / f32(trades - 1u);
        if (variance > 0.0) {
            sharpe = meanR / sqrt(variance);
        }
    }

    results.data[comboIndex] = Result(
        finalEquity: equity,
        profitFactor: pf,
        sharpe: sharpe,
        maxDrawdown: maxDD,
        numTrades: trades,
        winTrades: winTrades,
        lossTrades: lossTrades,
        totalTradeBars: totalTradeBars,
        barsWithPosition: barsWithPosition,
        _pad: 0u,
        _pad2: 0u
    );
}
