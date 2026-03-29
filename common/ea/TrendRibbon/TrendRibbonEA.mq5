//+------------------------------------------------------------------+
//| TrendRibbonEA.mq5 — Trend Ribbon M30+H4 Multi-Symbol EA         |
//|                                                                  |
//| Auto-detects Strategy Tester for single-symbol backtesting.      |
//+------------------------------------------------------------------+
#property copyright "TR AutoTrader"
#property version   "1.00"
#property strict

#include "Config.mqh"
#include "Logger.mqh"
#include "GridCalculator.mqh"
#include "SignalEngine.mqh"
#include "RiskManager.mqh"
#include "NewsFilter.mqh"

//--- Global objects
CLogger          g_logger;
CGridCalculator  g_grid;
CSignalEngine    g_signals;
CRiskManager     g_risk;
CNewsFilter      g_news;
bool             g_isTester = false;

// Parsed allowed entry hours
bool             g_allowedHours[24];
bool             g_useTimeFilter = false;

//+------------------------------------------------------------------+
//| Parse allowed hours string into boolean array                      |
//+------------------------------------------------------------------+
void ParseAllowedHours() {
   // Initialize all hours as allowed
   for(int i = 0; i < 24; i++)
      g_allowedHours[i] = true;

   string hours = InpAllowedHours;
   StringTrimLeft(hours);
   StringTrimRight(hours);

   if(StringLen(hours) == 0)
      return;  // No filter — all hours allowed

   g_useTimeFilter = true;

   // Block all, then enable specified hours
   for(int i = 0; i < 24; i++)
      g_allowedHours[i] = false;

   string parts[];
   int count = StringSplit(hours, ',', parts);
   for(int i = 0; i < count; i++) {
      StringTrimLeft(parts[i]);
      StringTrimRight(parts[i]);
      int h = (int)StringToInteger(parts[i]);
      if(h >= 0 && h < 24)
         g_allowedHours[h] = true;
   }
}

//+------------------------------------------------------------------+
//| Check if current server hour allows entry                          |
//+------------------------------------------------------------------+
bool IsEntryHourAllowed() {
   if(!g_useTimeFilter) return true;
   MqlDateTime dt;
   TimeCurrent(dt);
   return g_allowedHours[dt.hour];
}

//+------------------------------------------------------------------+
//| Check if current spread is acceptable for entry                    |
//+------------------------------------------------------------------+
bool IsSpreadAllowed(string sym) {
   if(InpMaxSpreadPts <= 0) return true;
   long spread = SymbolInfoInteger(sym, SYMBOL_SPREAD);
   return (spread <= InpMaxSpreadPts);
}

//+------------------------------------------------------------------+
//| Expert initialization                                             |
//+------------------------------------------------------------------+
int OnInit() {
   // Auto-detect Strategy Tester mode
   g_isTester = (bool)MQLInfoInteger(MQL_TESTER);

   // Parse symbol configuration
   if(!ParseSymbols()) {
      Print("[EA] Failed to parse symbols");
      return INIT_FAILED;
   }

   // Single-symbol mode: override to chart symbol only (for Strategy Tester)
   if(g_isTester) {
      g_symbolCount = 1;
      g_symbols[0].name = _Symbol;
      // Re-detect symbol properties
      g_symbols[0].digits = (int)SymbolInfoInteger(_Symbol, SYMBOL_DIGITS);
      double point = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
      if(g_symbols[0].digits == 3 || g_symbols[0].digits == 5)
         g_symbols[0].pipSize = point * 10;
      else if(_Symbol == "XAUUSD")
         g_symbols[0].pipSize = 0.10;
      else
         g_symbols[0].pipSize = point;

      if(StringFind(_Symbol, "JPY") >= 0) {
         g_symbols[0].quoteCcy = "JPY";
         g_symbols[0].pipValuePerLot = 7.0;
      } else {
         g_symbols[0].quoteCcy = "USD";
         g_symbols[0].pipValuePerLot = 10.0;
      }
      // Use the lot size matching the symbol or default
      if(_Symbol == "EURUSD")      g_symbols[0].lotSize = InpLotEURUSD;
      else if(_Symbol == "USDJPY") g_symbols[0].lotSize = InpLotUSDJPY;
      else if(_Symbol == "EURJPY") g_symbols[0].lotSize = InpLotEURJPY;
      else if(_Symbol == "XAUUSD") g_symbols[0].lotSize = InpLotXAUUSD;
      else if(_Symbol == "GBPUSD") g_symbols[0].lotSize = InpLotGBPUSD;
      else                         g_symbols[0].lotSize = 0.1;
   }

   // Parse entry time filter
   ParseAllowedHours();

   // Initialize components
   if(!g_logger.Init("TrendRibbon.log", !MQLInfoInteger(MQL_TESTER))) {
      Print("[EA] Logger init failed");
   }

   if(!g_grid.Init(InpEntryTF, InpFilterTF)) {
      g_logger.Error("GridCalculator init failed");
      return INIT_FAILED;
   }

   if(!g_signals.Init(&g_grid, &g_logger)) {
      g_logger.Error("SignalEngine init failed");
      return INIT_FAILED;
   }

   if(!g_risk.Init(&g_logger)) {
      g_logger.Error("RiskManager init failed");
      return INIT_FAILED;
   }

   if(!g_news.Init(&g_logger)) {
      g_logger.Error("NewsFilter init failed");
   }

   // Start timer for multi-symbol polling
   EventSetTimer(InpTimerSec);

   // Log startup
   g_logger.Info("============================================================");
   g_logger.Info(StringFormat("  Trend Ribbon M30+H4 EA v1.00 [%s]",
                 g_isTester ? "SINGLE/TESTER" : "MULTI-SYMBOL"));
   string symList = "";
   for(int i = 0; i < g_symbolCount; i++) {
      if(i > 0) symList += ", ";
      symList += g_symbols[i].name + "(" + DoubleToString(g_symbols[i].lotSize, 2) + ")";
   }
   g_logger.Info("  Symbols: " + symList);
   g_logger.Info(StringFormat("  Balance: $%.2f  Equity: $%.2f",
                 AccountInfoDouble(ACCOUNT_BALANCE),
                 AccountInfoDouble(ACCOUNT_EQUITY)));
   g_logger.Info(StringFormat("  Emergency SL budget: $%.0f/symbol", InpSLBudgetUSD));
   g_logger.Info(StringFormat("  News filter: %s", InpNewsFilter ? "ON" : "OFF"));
   g_logger.Info(StringFormat("  Relaxed entry: %s", InpRelaxedEntry ? "ON" : "OFF"));
   g_logger.Info(StringFormat("  Entry hours: %s", g_useTimeFilter ? InpAllowedHours : "ALL"));
   g_logger.Info("============================================================");

   return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
//| Expert deinitialization                                           |
//+------------------------------------------------------------------+
void OnDeinit(const int reason) {
   EventKillTimer();
   g_grid.Deinit();
   g_logger.Info("EA stopped (reason=" + IntegerToString(reason) + ")");
   g_logger.Deinit();
}

//+------------------------------------------------------------------+
//| Timer event — main polling loop for multi-symbol                  |
//+------------------------------------------------------------------+
void OnTimer() {
   ProcessAllSymbols();
}

//+------------------------------------------------------------------+
//| Tick event — used in tester mode                                  |
//+------------------------------------------------------------------+
void OnTick() {
   static int tickCount = 0;
   tickCount++;
   if(tickCount <= 3) {
      PrintFormat("[TICK#%d] g_isTester=%d, symbolCount=%d, symbol=%s",
                  tickCount, g_isTester, g_symbolCount,
                  g_symbolCount > 0 ? g_symbols[0].name : "NONE");
      PrintFormat("[TICK#%d] M30 bars=%d, H4 bars=%d, warmedUp_M30=%d, warmedUp_H4=%d",
                  tickCount,
                  Bars(_Symbol, InpEntryTF),
                  Bars(_Symbol, InpFilterTF),
                  g_symbolCount > 0 ? g_grid.IsWarmedUp(0, 0) : -1,
                  g_symbolCount > 0 ? g_grid.IsWarmedUp(0, 1) : -1);
   }
   if(g_isTester) {
      ProcessAllSymbols();
   }
}

//+------------------------------------------------------------------+
//| Process all symbols                                               |
//+------------------------------------------------------------------+
void ProcessAllSymbols() {
   // Risk check
   g_risk.Update();

   if(g_risk.ShouldCloseAll()) {
      CloseAllPositions();
      return;
   }

   // Process each symbol
   for(int i = 0; i < g_symbolCount; i++) {
      ProcessSymbol(i);
   }
}

//+------------------------------------------------------------------+
//| Process one symbol                                                |
//+------------------------------------------------------------------+
void ProcessSymbol(int symIdx) {
   string sym = g_symbols[symIdx].name;

   // Get current position direction for this symbol/magic
   int currentDir = GetPositionDirection(sym);

   // Update signal engine
   ENUM_ACTION action = g_signals.Update(symIdx, currentDir);

   if(action == ACTION_NONE)
      return;

   // Execute
   switch(action) {
      case ACTION_EXIT:
         DoExit(symIdx);
         break;

      case ACTION_ENTER_LONG:
         if(g_risk.CanEnter() && g_news.CanEnter(sym) && IsEntryHourAllowed() && IsSpreadAllowed(sym))
            DoEnter(symIdx, 1);
         break;

      case ACTION_ENTER_SHORT:
         if(g_risk.CanEnter() && g_news.CanEnter(sym) && IsEntryHourAllowed() && IsSpreadAllowed(sym))
            DoEnter(symIdx, -1);
         break;

      case ACTION_REVERSE_LONG:
         DoExit(symIdx);
         if(g_risk.CanEnter() && g_news.CanEnter(sym) && IsEntryHourAllowed() && IsSpreadAllowed(sym))
            DoEnter(symIdx, 1);
         break;

      case ACTION_REVERSE_SHORT:
         DoExit(symIdx);
         if(g_risk.CanEnter() && g_news.CanEnter(sym) && IsEntryHourAllowed() && IsSpreadAllowed(sym))
            DoEnter(symIdx, -1);
         break;
   }
}

//+------------------------------------------------------------------+
//| Place market order with emergency SL                              |
//+------------------------------------------------------------------+
void DoEnter(int symIdx, int direction) {
   string sym = g_symbols[symIdx].name;
   double lot = g_symbols[symIdx].lotSize;

   // Get current price
   double ask = SymbolInfoDouble(sym, SYMBOL_ASK);
   double bid = SymbolInfoDouble(sym, SYMBOL_BID);
   double price = (direction == 1) ? ask : bid;

   if(price <= 0) {
      g_logger.Error(StringFormat("Cannot get price for %s", sym));
      return;
   }

   // Recalculate pip value for JPY pairs
   double pipVal = g_symbols[symIdx].pipValuePerLot;
   if(g_symbols[symIdx].quoteCcy == "JPY") {
      double usdjpyBid = SymbolInfoDouble("USDJPY", SYMBOL_BID);
      if(usdjpyBid > 0)
         pipVal = 100000.0 * 0.01 / usdjpyBid;
   }

   // Emergency SL
   double sl = g_risk.CalcEmergencySL(direction, price, lot,
               g_symbols[symIdx].pipSize, pipVal);

   // Clamp SL to valid range (must be positive, set 0 to disable if invalid)
   if(sl <= 0) sl = 0;

   double slPips = (sl > 0) ? MathAbs(price - sl) / g_symbols[symIdx].pipSize : 0;

   // Determine filling mode
   ENUM_ORDER_TYPE_FILLING filling = GetFillingMode(sym);

   // Build trade request
   MqlTradeRequest request = {};
   MqlTradeResult  result  = {};

   request.action    = TRADE_ACTION_DEAL;
   request.symbol    = sym;
   request.volume    = lot;
   request.type      = (direction == 1) ? ORDER_TYPE_BUY : ORDER_TYPE_SELL;
   request.price     = price;
   request.sl        = sl;
   request.deviation = InpSlippage;
   request.magic     = InpMagicNumber;
   request.comment   = InpComment;
   request.type_filling = filling;

   string dirStr = (direction == 1) ? "LONG" : "SHORT";

   if(!OrderSend(request, result)) {
      g_logger.Error(StringFormat("ENTRY FAILED %s %s %.2f lots: err=%d, retcode=%d",
                     dirStr, sym, lot, GetLastError(), result.retcode));
      return;
   }

   if(result.retcode == TRADE_RETCODE_DONE || result.retcode == TRADE_RETCODE_PLACED) {
      g_logger.Info(StringFormat("ENTRY %s %s %.2f lots @ %.5f, SL=%.5f (%.0f pips), ticket=%d",
                    dirStr, sym, lot, result.price, sl, slPips, result.deal));
   } else {
      g_logger.Error(StringFormat("ENTRY %s %s retcode=%d: %s",
                     dirStr, sym, result.retcode, result.comment));
   }
}

//+------------------------------------------------------------------+
//| Close position for symbol                                         |
//+------------------------------------------------------------------+
void DoExit(int symIdx) {
   string sym = g_symbols[symIdx].name;

   for(int i = PositionsTotal() - 1; i >= 0; i--) {
      if(PositionGetSymbol(i) != sym) continue;
      if(PositionGetInteger(POSITION_MAGIC) != InpMagicNumber) continue;

      ulong ticket = PositionGetInteger(POSITION_TICKET);
      double volume = PositionGetDouble(POSITION_VOLUME);
      int type = (int)PositionGetInteger(POSITION_TYPE);

      MqlTradeRequest request = {};
      MqlTradeResult  result  = {};

      request.action   = TRADE_ACTION_DEAL;
      request.symbol   = sym;
      request.volume   = volume;
      request.type     = (type == POSITION_TYPE_BUY) ? ORDER_TYPE_SELL : ORDER_TYPE_BUY;
      request.price    = (type == POSITION_TYPE_BUY) ?
                         SymbolInfoDouble(sym, SYMBOL_BID) :
                         SymbolInfoDouble(sym, SYMBOL_ASK);
      request.position = ticket;
      request.deviation = InpSlippage;
      request.magic    = InpMagicNumber;
      request.comment  = InpComment;
      request.type_filling = GetFillingMode(sym);

      if(!OrderSend(request, result)) {
         g_logger.Error(StringFormat("EXIT FAILED %s ticket=%d: err=%d",
                        sym, ticket, GetLastError()));
      } else if(result.retcode == TRADE_RETCODE_DONE) {
         double profit = PositionGetDouble(POSITION_PROFIT);
         g_logger.Info(StringFormat("EXIT %s ticket=%d, profit=%.2f",
                       sym, ticket, profit));
      } else {
         g_logger.Error(StringFormat("EXIT %s retcode=%d: %s",
                        sym, result.retcode, result.comment));
      }
   }
}

//+------------------------------------------------------------------+
//| Close ALL positions for this magic number                         |
//+------------------------------------------------------------------+
void CloseAllPositions() {
   g_logger.Critical("CLOSING ALL POSITIONS - FTMO risk limit");
   for(int i = 0; i < g_symbolCount; i++)
      DoExit(i);
}

//+------------------------------------------------------------------+
//| Get current position direction for symbol (1=long, -1=short, 0)  |
//+------------------------------------------------------------------+
int GetPositionDirection(string sym) {
   for(int i = PositionsTotal() - 1; i >= 0; i--) {
      if(PositionGetSymbol(i) != sym) continue;
      if(PositionGetInteger(POSITION_MAGIC) != InpMagicNumber) continue;
      int type = (int)PositionGetInteger(POSITION_TYPE);
      return (type == POSITION_TYPE_BUY) ? 1 : -1;
   }
   return 0;
}

//+------------------------------------------------------------------+
//| Determine filling mode for symbol                                 |
//+------------------------------------------------------------------+
ENUM_ORDER_TYPE_FILLING GetFillingMode(string sym) {
   long modes = SymbolInfoInteger(sym, SYMBOL_FILLING_MODE);
   if((modes & SYMBOL_FILLING_FOK) != 0) return ORDER_FILLING_FOK;
   if((modes & SYMBOL_FILLING_IOC) != 0) return ORDER_FILLING_IOC;
   return ORDER_FILLING_RETURN;
}
//+------------------------------------------------------------------+
