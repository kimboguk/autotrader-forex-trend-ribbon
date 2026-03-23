//+------------------------------------------------------------------+
//| NewsFilter.mqh — MQL5 built-in economic calendar news filter     |
//+------------------------------------------------------------------+
#ifndef NEWS_FILTER_MQH
#define NEWS_FILTER_MQH

#include "Config.mqh"
#include "Logger.mqh"

class CNewsFilter {
private:
   CLogger *m_log;
   bool     m_enabled;
   int      m_beforeMin;
   int      m_afterMin;

   // Currency to symbol mapping
   string GetCurrenciesForSymbol(string sym) {
      // Return comma-separated currencies affected by this symbol
      if(sym == "EURUSD") return "EUR,USD";
      if(sym == "USDJPY") return "USD,JPY";
      if(sym == "EURJPY") return "EUR,JPY";
      if(sym == "GBPUSD") return "GBP,USD";
      if(sym == "XAUUSD") return "USD";  // Gold mainly affected by USD events
      // Generic: extract base and quote
      string base = StringSubstr(sym, 0, 3);
      string quote = StringSubstr(sym, 3, 3);
      return base + "," + quote;
   }

public:
   CNewsFilter() : m_log(NULL), m_enabled(false) {}

   bool Init(CLogger *log) {
      m_log = log;
      m_enabled = InpNewsFilter;
      m_beforeMin = InpNewsBeforeMin;
      m_afterMin  = InpNewsAfterMin;
      return true;
   }

   bool CanEnter(string symbol) {
      if(!m_enabled) return true;

      string currencies = GetCurrenciesForSymbol(symbol);
      string parts[];
      int count = StringSplit(currencies, ',', parts);

      datetime now = TimeCurrent();
      datetime from = now - m_beforeMin * 60;
      datetime to   = now + m_afterMin * 60;

      // Use MQL5 built-in calendar
      MqlCalendarValue values[];

      for(int i = 0; i < count; i++) {
         StringTrimLeft(parts[i]);
         StringTrimRight(parts[i]);

         // Get country code from currency
         string country = CurrencyToCountry(parts[i]);
         if(country == "") continue;

         int total = CalendarValueHistory(values, from, to, country);
         if(total <= 0) continue;

         for(int v = 0; v < total; v++) {
            // Get event details to check importance
            MqlCalendarEvent event;
            if(!CalendarEventById(values[v].event_id, event))
               continue;

            // Only block on high-impact events
            if(event.importance == CALENDAR_IMPORTANCE_HIGH) {
               m_log.Warning(StringFormat("NEWS BLOCK %s: %s event at %s (currency=%s)",
                             symbol, event.name,
                             TimeToString(values[v].time, TIME_MINUTES),
                             parts[i]));
               return false;
            }
         }
      }

      return true;
   }

private:
   string CurrencyToCountry(string ccy) {
      if(ccy == "USD") return "US";
      if(ccy == "EUR") return "EU";
      if(ccy == "GBP") return "GB";
      if(ccy == "JPY") return "JP";
      if(ccy == "AUD") return "AU";
      if(ccy == "NZD") return "NZ";
      if(ccy == "CAD") return "CA";
      if(ccy == "CHF") return "CH";
      return "";
   }
};

#endif
