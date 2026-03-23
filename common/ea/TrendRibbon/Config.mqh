//+------------------------------------------------------------------+
//| Config.mqh — Input parameters and symbol configuration           |
//+------------------------------------------------------------------+
#ifndef CONFIG_MQH
#define CONFIG_MQH

//--- Strategy inputs
input group "=== Strategy ==="
input ENUM_TIMEFRAMES InpEntryTF    = PERIOD_M30;  // Entry timeframe
input ENUM_TIMEFRAMES InpFilterTF   = PERIOD_H4;   // H4 filter timeframe
input int    InpEMA1        = 30;    // EMA period 1
input int    InpEMA2        = 60;    // EMA period 2
input int    InpEMA3        = 120;   // EMA period 3
input int    InpEMA4        = 240;   // EMA period 4
input bool   InpRelaxedEntry = true; // Relaxed entry condition
input int    InpBarCount    = 500;   // Bars to load for analysis

//--- Symbols (comma-separated)
input group "=== Symbols & Lots ==="
input string InpSymbols     = "EURUSD,USDJPY,EURJPY,XAUUSD,GBPUSD";
input double InpLotEURUSD   = 1.0;
input double InpLotUSDJPY   = 1.0;
input double InpLotEURJPY   = 1.0;
input double InpLotXAUUSD   = 1.0;
input double InpLotGBPUSD   = 1.0;

//--- FTMO Risk
input group "=== FTMO Risk ==="
input double InpAccountSize     = 200000; // Account size
input double InpBlockDailyPct   = 4.0;    // Block entries at daily loss %
input double InpCloseDailyPct   = 4.5;    // Force-close at daily loss %
input double InpBlockTotalPct   = 8.0;    // Block entries at total DD %
input double InpCloseTotalPct   = 9.0;    // Force-close at total DD %
input double InpSLBudgetUSD    = 2000;   // Emergency SL budget per symbol

//--- News Filter
input group "=== News Filter ==="
input bool   InpNewsFilter      = true;   // Enable news filter
input int    InpNewsBeforeMin   = 2;      // Minutes before event
input int    InpNewsAfterMin    = 2;      // Minutes after event

//--- Execution
input group "=== Execution ==="
input int    InpMagicNumber     = 20260319; // Magic number
input string InpComment         = "TR_M30H4"; // Order comment
input int    InpSlippage        = 10;     // Max slippage points
input int    InpTimerSec        = 5;      // Poll interval seconds
// Single-symbol mode is auto-detected via MQL_TESTER

//--- Constants
#define MAX_SYMBOLS    10
#define MAX_EMA        4

//--- Symbol config structure
struct SymbolConfig {
   string name;
   double lotSize;
   double pipSize;
   double pipValuePerLot;  // approx USD per pip per lot
   string quoteCcy;
   int    digits;
};

//--- Global symbol array
SymbolConfig g_symbols[];
int          g_symbolCount = 0;
int          g_emaPeriods[MAX_EMA];

//+------------------------------------------------------------------+
//| Parse symbol list and populate g_symbols[]                        |
//+------------------------------------------------------------------+
bool ParseSymbols() {
   g_emaPeriods[0] = InpEMA1;
   g_emaPeriods[1] = InpEMA2;
   g_emaPeriods[2] = InpEMA3;
   g_emaPeriods[3] = InpEMA4;

   string parts[];
   int count = StringSplit(InpSymbols, ',', parts);
   if(count <= 0) return false;

   ArrayResize(g_symbols, count);
   g_symbolCount = count;

   // Lot size lookup
   double lots[];
   ArrayResize(lots, count);
   for(int i = 0; i < count; i++) {
      StringTrimLeft(parts[i]);
      StringTrimRight(parts[i]);
      // Default lot mapping
      if(parts[i] == "EURUSD")      lots[i] = InpLotEURUSD;
      else if(parts[i] == "USDJPY") lots[i] = InpLotUSDJPY;
      else if(parts[i] == "EURJPY") lots[i] = InpLotEURJPY;
      else if(parts[i] == "XAUUSD") lots[i] = InpLotXAUUSD;
      else if(parts[i] == "GBPUSD") lots[i] = InpLotGBPUSD;
      else                          lots[i] = 0.1;
   }

   for(int i = 0; i < count; i++) {
      g_symbols[i].name = parts[i];
      g_symbols[i].lotSize = lots[i];

      // Ensure symbol is in Market Watch
      if(!SymbolSelect(parts[i], true)) {
         PrintFormat("[Config] WARNING: Cannot select %s in Market Watch", parts[i]);
      }

      g_symbols[i].digits = (int)SymbolInfoInteger(parts[i], SYMBOL_DIGITS);

      // Pip size / pip value
      double point = SymbolInfoDouble(parts[i], SYMBOL_POINT);
      if(g_symbols[i].digits == 3 || g_symbols[i].digits == 5)
         g_symbols[i].pipSize = point * 10;
      else if(parts[i] == "XAUUSD")
         g_symbols[i].pipSize = 0.10;
      else
         g_symbols[i].pipSize = point;

      // Quote currency and pip value approximation
      if(StringFind(parts[i], "JPY") >= 0) {
         g_symbols[i].quoteCcy = "JPY";
         g_symbols[i].pipValuePerLot = 7.0;  // approx, recalc at runtime
      } else if(parts[i] == "XAUUSD") {
         g_symbols[i].quoteCcy = "USD";
         g_symbols[i].pipValuePerLot = 10.0;  // 100oz × $0.10
      } else {
         g_symbols[i].quoteCcy = "USD";
         g_symbols[i].pipValuePerLot = 10.0;  // standard lot
      }
   }

   return true;
}

#endif
