//+------------------------------------------------------------------+
//| RiskManager.mqh — FTMO daily/total drawdown protection           |
//+------------------------------------------------------------------+
#ifndef RISK_MANAGER_MQH
#define RISK_MANAGER_MQH

#include "Config.mqh"
#include "Logger.mqh"

class CRiskManager {
private:
   CLogger *m_log;

   double m_accountSize;
   double m_blockDaily;    // block entries threshold
   double m_closeDaily;    // force-close threshold
   double m_blockTotal;
   double m_closeTotal;

   double m_dailyStartEquity;
   string m_currentDate;

   bool   m_entriesBlocked;
   bool   m_forceClose;

public:
   CRiskManager() : m_log(NULL), m_entriesBlocked(false), m_forceClose(false),
                     m_dailyStartEquity(0), m_currentDate("") {}

   bool Init(CLogger *log) {
      m_log = log;
      m_accountSize = InpAccountSize;
      m_blockDaily  = m_accountSize * InpBlockDailyPct / 100.0;
      m_closeDaily  = m_accountSize * InpCloseDailyPct / 100.0;
      m_blockTotal  = m_accountSize * InpBlockTotalPct / 100.0;
      m_closeTotal  = m_accountSize * InpCloseTotalPct / 100.0;
      return true;
   }

   void Update() {
      double equity  = AccountInfoDouble(ACCOUNT_EQUITY);
      double balance = AccountInfoDouble(ACCOUNT_BALANCE);

      // Daily reset
      string today = TimeToString(TimeCurrent(), TIME_DATE);
      if(m_currentDate != today) {
         m_dailyStartEquity = MathMax(balance, equity);
         m_currentDate = today;
         m_entriesBlocked = false;
         m_forceClose = false;
         m_log.Info(StringFormat("Daily reset: start_equity=%.2f, date=%s",
                    m_dailyStartEquity, today));
      }

      if(m_dailyStartEquity <= 0)
         m_dailyStartEquity = MathMax(balance, equity);

      double dailyLoss = m_dailyStartEquity - equity;
      double totalDD   = m_accountSize - equity;

      // Force-close check
      if(dailyLoss >= m_closeDaily || totalDD >= m_closeTotal) {
         m_forceClose = true;
         m_entriesBlocked = true;
         m_log.Critical(StringFormat("FORCE CLOSE - daily_loss=%.2f/%.2f, total_dd=%.2f/%.2f",
                        dailyLoss, m_closeDaily, totalDD, m_closeTotal));
         return;
      }

      // Entry-block check
      if(dailyLoss >= m_blockDaily || totalDD >= m_blockTotal) {
         m_entriesBlocked = true;
         m_log.Warning(StringFormat("Entries BLOCKED - daily_loss=%.2f/%.2f, total_dd=%.2f/%.2f",
                       dailyLoss, m_blockDaily, totalDD, m_blockTotal));
         return;
      }

      m_entriesBlocked = false;
      m_forceClose = false;
   }

   bool CanEnter()       { return !m_entriesBlocked; }
   bool ShouldCloseAll() { return m_forceClose; }

   //--- Emergency SL calculation ---
   double CalcEmergencySL(int direction, double entryPrice,
                          double lotSize, double pipSize, double pipValuePerLot) {
      double maxPips = InpSLBudgetUSD / (lotSize * pipValuePerLot);
      double slDist  = maxPips * pipSize;

      if(direction == 1)
         return NormalizeDouble(entryPrice - slDist, 5);
      else
         return NormalizeDouble(entryPrice + slDist, 5);
   }
};

#endif
