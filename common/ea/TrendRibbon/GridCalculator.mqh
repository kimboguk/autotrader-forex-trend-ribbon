//+------------------------------------------------------------------+
//| GridCalculator.mqh — EMA grid computation via iMA() handles      |
//+------------------------------------------------------------------+
#ifndef GRID_CALCULATOR_MQH
#define GRID_CALCULATOR_MQH

#include "Config.mqh"
#include "Logger.mqh"

//--- Grid values for one bar
struct GridValues {
   double gridTop;
   double gridBottom;
   double bodyMid;
   bool   isBullish;
   double open;
   double close;
   bool   valid;
};

class CGridCalculator {
private:
   // iMA handles: [symbolIdx][emaIdx][tfIdx: 0=entry, 1=filter]
   int m_handles[MAX_SYMBOLS][MAX_EMA][2];
   int m_symbolCount;
   ENUM_TIMEFRAMES m_entryTF;
   ENUM_TIMEFRAMES m_filterTF;

public:
   CGridCalculator() : m_symbolCount(0) {}

   bool Init(ENUM_TIMEFRAMES entryTF, ENUM_TIMEFRAMES filterTF) {
      m_entryTF = entryTF;
      m_filterTF = filterTF;
      m_symbolCount = g_symbolCount;

      // Initialize all handles to INVALID
      for(int s = 0; s < MAX_SYMBOLS; s++)
         for(int e = 0; e < MAX_EMA; e++)
            for(int t = 0; t < 2; t++)
               m_handles[s][e][t] = INVALID_HANDLE;

      // Create iMA handles
      for(int s = 0; s < m_symbolCount; s++) {
         for(int e = 0; e < MAX_EMA; e++) {
            m_handles[s][e][0] = iMA(g_symbols[s].name, entryTF,
                                      g_emaPeriods[e], 0, MODE_EMA, PRICE_CLOSE);
            m_handles[s][e][1] = iMA(g_symbols[s].name, filterTF,
                                      g_emaPeriods[e], 0, MODE_EMA, PRICE_CLOSE);

            if(m_handles[s][e][0] == INVALID_HANDLE ||
               m_handles[s][e][1] == INVALID_HANDLE) {
               PrintFormat("[Grid] Failed to create iMA handle: %s period=%d (err=%d)",
                           g_symbols[s].name, g_emaPeriods[e], GetLastError());
               return false;
            }
         }
      }
      return true;
   }

   void Deinit() {
      for(int s = 0; s < m_symbolCount; s++)
         for(int e = 0; e < MAX_EMA; e++)
            for(int t = 0; t < 2; t++)
               if(m_handles[s][e][t] != INVALID_HANDLE)
                  IndicatorRelease(m_handles[s][e][t]);
   }

   //--- Get grid values for a completed bar
   //    shift: 1 = last completed bar, 2 = one before that
   //    tfIdx: 0 = entry TF, 1 = filter TF
   bool GetGridValues(int symIdx, int tfIdx, int shift, GridValues &gv) {
      gv.valid = false;

      ENUM_TIMEFRAMES tf = (tfIdx == 0) ? m_entryTF : m_filterTF;
      string sym = g_symbols[symIdx].name;

      double emas[MAX_EMA];
      double buf[1];

      for(int e = 0; e < MAX_EMA; e++) {
         if(CopyBuffer(m_handles[symIdx][e][tfIdx], 0, shift, 1, buf) != 1)
            return false;
         emas[e] = buf[0];
      }

      // grid_top = max(emas), grid_bottom = min(emas)
      gv.gridTop = emas[0];
      gv.gridBottom = emas[0];
      for(int e = 1; e < MAX_EMA; e++) {
         if(emas[e] > gv.gridTop)    gv.gridTop = emas[e];
         if(emas[e] < gv.gridBottom) gv.gridBottom = emas[e];
      }

      // OHLC of the bar
      gv.open  = iOpen(sym, tf, shift);
      gv.close = iClose(sym, tf, shift);
      gv.bodyMid = (gv.open + gv.close) / 2.0;
      gv.isBullish = (gv.close >= gv.open);
      gv.valid = true;

      return true;
   }

   //--- Get bar open time at shift
   datetime GetBarTime(int symIdx, int tfIdx, int shift) {
      ENUM_TIMEFRAMES tf = (tfIdx == 0) ? m_entryTF : m_filterTF;
      return iTime(g_symbols[symIdx].name, tf, shift);
   }

   ENUM_TIMEFRAMES GetFilterTF() { return m_filterTF; }

   //--- Check if enough bars are available for warmup
   bool IsWarmedUp(int symIdx, int tfIdx) {
      ENUM_TIMEFRAMES tf = (tfIdx == 0) ? m_entryTF : m_filterTF;
      int bars = Bars(g_symbols[symIdx].name, tf);
      int maxPeriod = g_emaPeriods[MAX_EMA - 1]; // 240
      return (bars > maxPeriod + 10);
   }
};

#endif
