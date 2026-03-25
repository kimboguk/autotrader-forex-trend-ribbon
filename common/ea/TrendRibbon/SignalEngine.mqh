//+------------------------------------------------------------------+
//| SignalEngine.mqh — M30 entry/exit signals with H4 filter         |
//+------------------------------------------------------------------+
#ifndef SIGNAL_ENGINE_MQH
#define SIGNAL_ENGINE_MQH

#include "Config.mqh"
#include "GridCalculator.mqh"
#include "Logger.mqh"

//--- Signal codes (same as Python)
#define SIG_NONE           0
#define SIG_ENTER_LONG     1
#define SIG_ENTER_SHORT   -1
#define SIG_EXIT_LONG    -10
#define SIG_EXIT_SHORT    10
#define SIG_REVERSE_LONG   2
#define SIG_REVERSE_SHORT -2

//--- Action strings
enum ENUM_ACTION {
   ACTION_NONE,
   ACTION_ENTER_LONG,
   ACTION_ENTER_SHORT,
   ACTION_EXIT,
   ACTION_REVERSE_LONG,
   ACTION_REVERSE_SHORT
};

class CSignalEngine {
private:
   CGridCalculator *m_grid;
   CLogger         *m_log;

   // H4 directional filter per symbol: 1=long, -1=short, 0=flat
   int m_h4Position[MAX_SYMBOLS];

   // Last processed bar times
   datetime m_lastM30Bar[MAX_SYMBOLS];
   datetime m_lastH4Bar[MAX_SYMBOLS];

public:
   CSignalEngine() : m_grid(NULL), m_log(NULL) {}

   bool Init(CGridCalculator *grid, CLogger *log) {
      m_grid = grid;
      m_log = log;
      for(int i = 0; i < MAX_SYMBOLS; i++) {
         m_h4Position[i] = 0;
         m_lastM30Bar[i] = 0;
         m_lastH4Bar[i] = 0;
      }
      return true;
   }

   //--- Main update: returns action for the symbol
   ENUM_ACTION Update(int symIdx, int currentPosition) {
      if(!m_grid.IsWarmedUp(symIdx, 0) || !m_grid.IsWarmedUp(symIdx, 1))
         return ACTION_NONE;

      // --- H4 filter update on new H4 bar ---
      datetime h4Time = m_grid.GetBarTime(symIdx, 1, 1); // completed H4 bar
      if(h4Time != m_lastH4Bar[symIdx]) {
         UpdateH4Position(symIdx);
         m_lastH4Bar[symIdx] = h4Time;
      }

      // --- Check for new M30 bar ---
      datetime m30Time = m_grid.GetBarTime(symIdx, 0, 1); // completed M30 bar
      if(m30Time == m_lastM30Bar[symIdx])
         return ACTION_NONE; // no new bar

      m_lastM30Bar[symIdx] = m30Time;

      // --- DEBUG: log every 1000th M30 bar ---
      static int dbgCount = 0;
      dbgCount++;
      if(dbgCount <= 3 || dbgCount % 1000 == 0) {
         GridValues dbgCurr, dbgPrev;
         m_grid.GetGridValues(symIdx, 0, 1, dbgCurr);
         m_grid.GetGridValues(symIdx, 0, 2, dbgPrev);
         PrintFormat("[DBG#%d] %s M30=%s H4pos=%+d pos=%d | gridTop=%.5f gridBot=%.5f bodyMid=%.5f bull=%d | prev_bm=%.5f prev_gt=%.5f",
                     dbgCount, g_symbols[symIdx].name,
                     TimeToString(m30Time, TIME_DATE|TIME_MINUTES),
                     m_h4Position[symIdx], currentPosition,
                     dbgCurr.gridTop, dbgCurr.gridBottom, dbgCurr.bodyMid, dbgCurr.isBullish,
                     dbgPrev.bodyMid, dbgPrev.gridTop);
      }

      // --- Compute signal ---
      int signal = CheckSignal(symIdx, currentPosition);
      if(signal == SIG_NONE)
         return ACTION_NONE;

      ENUM_ACTION action = SignalToAction(signal);

      // --- Apply H4 filter (entries only, exits always allowed) ---
      if(action == ACTION_ENTER_LONG || action == ACTION_REVERSE_LONG) {
         if(m_h4Position[symIdx] != 1) {
            m_log.Info(StringFormat("%s %s blocked by H4 filter (H4=%+d)",
                       g_symbols[symIdx].name,
                       action == ACTION_ENTER_LONG ? "enter_long" : "reverse_long",
                       m_h4Position[symIdx]));
            if(action == ACTION_REVERSE_LONG)
               return ACTION_EXIT;
            return ACTION_NONE;
         }
      }

      if(action == ACTION_ENTER_SHORT || action == ACTION_REVERSE_SHORT) {
         if(m_h4Position[symIdx] != -1) {
            m_log.Info(StringFormat("%s %s blocked by H4 filter (H4=%+d)",
                       g_symbols[symIdx].name,
                       action == ACTION_ENTER_SHORT ? "enter_short" : "reverse_short",
                       m_h4Position[symIdx]));
            if(action == ACTION_REVERSE_SHORT)
               return ACTION_EXIT;
            return ACTION_NONE;
         }
      }

      string actStr;
      switch(action) {
         case ACTION_ENTER_LONG:     actStr = "enter_long"; break;
         case ACTION_ENTER_SHORT:    actStr = "enter_short"; break;
         case ACTION_EXIT:           actStr = "exit"; break;
         case ACTION_REVERSE_LONG:   actStr = "reverse_long"; break;
         case ACTION_REVERSE_SHORT:  actStr = "reverse_short"; break;
         default: actStr = "none";
      }

      m_log.Info(StringFormat("SIGNAL %s: %s (H4=%+d, M30 bar=%s)",
                 g_symbols[symIdx].name, actStr, m_h4Position[symIdx],
                 TimeToString(m30Time, TIME_DATE | TIME_MINUTES)));

      return action;
   }

   int GetH4Position(int symIdx) { return m_h4Position[symIdx]; }

private:
   //--- Update H4 directional filter ---
   //    Uses same entry/exit logic as M30 CheckSignal, applied to H4 bars.
   //    On cold start (position=0), bootstraps from available H4 history.
   void UpdateH4Position(int symIdx) {
      int oldPos = m_h4Position[symIdx];

      // Cold start: iterate over H4 history to find current position
      if(oldPos == 0 && m_lastH4Bar[symIdx] == 0) {
         BootstrapH4Position(symIdx);
         return;
      }

      // Normal update: check last two completed H4 bars
      GridValues curr, prev;
      if(!m_grid.GetGridValues(symIdx, 1, 1, curr)) return;
      if(!m_grid.GetGridValues(symIdx, 1, 2, prev)) return;
      if(!curr.valid || !prev.valid) return;

      int newPos = ComputeNextPosition(oldPos, curr, prev);

      if(newPos != oldPos) {
         m_log.Info(StringFormat("H4 filter %s: %+d -> %+d",
                    g_symbols[symIdx].name, oldPos, newPos));
         PrintFormat("[H4] %s %s pos:%+d->%+d close=%.5f gt=%.5f gb=%.5f bm=%.5f bull=%d | prev_bm=%.5f prev_gt=%.5f",
                     g_symbols[symIdx].name,
                     TimeToString(m_grid.GetBarTime(symIdx, 1, 1), TIME_DATE|TIME_MINUTES),
                     oldPos, newPos, curr.close, curr.gridTop, curr.gridBottom,
                     curr.bodyMid, curr.isBullish, prev.bodyMid, prev.gridTop);
      }
      m_h4Position[symIdx] = newPos;
   }

   //--- Bootstrap H4 position from full history ---
   void BootstrapH4Position(int symIdx) {
      string sym = g_symbols[symIdx].name;
      int maxPeriod = g_emaPeriods[MAX_EMA - 1]; // 240

      // Get available H4 bar count
      int totalBars = Bars(sym, m_grid.GetFilterTF());
      if(totalBars < maxPeriod + 10) return;

      // Limit scan to last 500 H4 bars for performance
      int scanBars = MathMin(totalBars, 500);
      int position = 0;

      for(int shift = scanBars - 1; shift >= 2; shift--) {
         GridValues curr, prev;
         if(!m_grid.GetGridValues(symIdx, 1, shift, curr)) continue;
         if(!m_grid.GetGridValues(symIdx, 1, shift + 1, prev)) continue;
         if(!curr.valid || !prev.valid) continue;

         int newP = ComputeNextPosition(position, curr, prev);
         if(newP != position) {
            PrintFormat("[H4-BOOT] %s shift=%d pos:%+d->%+d close=%.5f gt=%.5f gb=%.5f bm=%.5f bull=%d",
                        sym, shift, position, newP,
                        curr.close, curr.gridTop, curr.gridBottom, curr.bodyMid, curr.isBullish);
         }
         position = newP;
      }

      if(position != m_h4Position[symIdx]) {
         m_log.Info(StringFormat("H4 filter %s: bootstrapped to %+d (scanned %d bars)",
                    sym, position, scanBars));
      }
      m_h4Position[symIdx] = position;
   }

   //--- Compute next position given current state and two bars ---
   int ComputeNextPosition(int position, GridValues &curr, GridValues &prev) {
      // Entry conditions (same as M30 CheckSignal)
      bool prevBelowTop = (prev.bodyMid <= prev.gridTop);
      bool prevAboveBot = (prev.bodyMid >= prev.gridBottom);

      // Relaxed entry
      if(InpRelaxedEntry) {
         if(!prev.isBullish && prev.close < prev.gridTop && prev.gridTop < prev.open)
            prevBelowTop = true;
         if(prev.isBullish && prev.open < prev.gridBottom && prev.gridBottom < prev.close)
            prevAboveBot = true;
      }

      bool longEntry  = curr.isBullish && (curr.bodyMid > curr.gridTop) && prevBelowTop;
      bool shortEntry = !curr.isBullish && (curr.bodyMid < curr.gridBottom) && prevAboveBot;
      bool longExit   = !curr.isBullish && (curr.bodyMid < curr.gridTop);
      bool shortExit  = curr.isBullish && (curr.bodyMid > curr.gridBottom);

      if(position == 0) {
         if(longEntry)       return 1;
         else if(shortEntry) return -1;
      }
      else if(position == 1) {
         if(longExit) {
            if(shortEntry) return -1;
            return 0;
         }
      }
      else if(position == -1) {
         if(shortExit) {
            if(longEntry) return 1;
            return 0;
         }
      }
      return position;
   }

   //--- Check M30 entry/exit signal ---
   int CheckSignal(int symIdx, int currentPosition) {
      GridValues curr, prev;
      if(!m_grid.GetGridValues(symIdx, 0, 1, curr)) return SIG_NONE;
      if(!m_grid.GetGridValues(symIdx, 0, 2, prev)) return SIG_NONE;
      if(!curr.valid || !prev.valid) return SIG_NONE;

      // Standard entry conditions
      bool prevBelowTop = (prev.bodyMid <= prev.gridTop);
      bool prevAboveBot = (prev.bodyMid >= prev.gridBottom);

      // Relaxed entry: prev bar straddled the grid boundary
      if(InpRelaxedEntry) {
         // Bearish candle straddling grid_top: close < grid_top < open
         if(!prev.isBullish && prev.close < prev.gridTop && prev.gridTop < prev.open)
            prevBelowTop = true;
         // Bullish candle straddling grid_bottom: open < grid_bottom < close
         if(prev.isBullish && prev.open < prev.gridBottom && prev.gridBottom < prev.close)
            prevAboveBot = true;
      }

      bool longEntry  = curr.isBullish && (curr.bodyMid > curr.gridTop) && prevBelowTop;
      bool shortEntry = !curr.isBullish && (curr.bodyMid < curr.gridBottom) && prevAboveBot;

      // Exit conditions
      bool longExit  = !curr.isBullish && (curr.bodyMid < curr.gridTop);
      bool shortExit = curr.isBullish && (curr.bodyMid > curr.gridBottom);

      if(currentPosition == 0) {
         if(longEntry)  return SIG_ENTER_LONG;
         if(shortEntry) return SIG_ENTER_SHORT;
      }
      else if(currentPosition == 1) {
         if(longExit) {
            if(shortEntry) return SIG_REVERSE_SHORT;
            return SIG_EXIT_LONG;
         }
      }
      else if(currentPosition == -1) {
         if(shortExit) {
            if(longEntry) return SIG_REVERSE_LONG;
            return SIG_EXIT_SHORT;
         }
      }

      return SIG_NONE;
   }

   ENUM_ACTION SignalToAction(int signal) {
      switch(signal) {
         case SIG_ENTER_LONG:    return ACTION_ENTER_LONG;
         case SIG_ENTER_SHORT:   return ACTION_ENTER_SHORT;
         case SIG_EXIT_LONG:     return ACTION_EXIT;
         case SIG_EXIT_SHORT:    return ACTION_EXIT;
         case SIG_REVERSE_LONG:  return ACTION_REVERSE_LONG;
         case SIG_REVERSE_SHORT: return ACTION_REVERSE_SHORT;
         default:                return ACTION_NONE;
      }
   }
};

#endif
