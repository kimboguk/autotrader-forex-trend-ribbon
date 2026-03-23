//+------------------------------------------------------------------+
//| Logger.mqh — File + journal logging utility                      |
//+------------------------------------------------------------------+
#ifndef LOGGER_MQH
#define LOGGER_MQH

class CLogger {
private:
   int    m_handle;
   bool   m_fileEnabled;
   string m_filename;

public:
   CLogger() : m_handle(INVALID_HANDLE), m_fileEnabled(false) {}

   bool Init(string filename = "TrendRibbon.log", bool fileLogging = true) {
      m_fileEnabled = fileLogging;
      m_filename = filename;
      if(m_fileEnabled) {
         m_handle = FileOpen("Logs/" + filename,
                             FILE_WRITE | FILE_TXT | FILE_SHARE_READ | FILE_ANSI,
                             0, CP_UTF8);
         if(m_handle == INVALID_HANDLE) {
            PrintFormat("[Logger] Cannot open log file: %s (err=%d)", filename, GetLastError());
            m_fileEnabled = false;
            return false;
         }
         // Seek to end for append
         FileSeek(m_handle, 0, SEEK_END);
      }
      return true;
   }

   void Deinit() {
      if(m_handle != INVALID_HANDLE) {
         FileClose(m_handle);
         m_handle = INVALID_HANDLE;
      }
   }

   void Info(string msg) {
      string line = TimeToString(TimeCurrent(), TIME_DATE | TIME_SECONDS) + " INFO  " + msg;
      Print(line);
      _WriteFile(line);
   }

   void Warning(string msg) {
      string line = TimeToString(TimeCurrent(), TIME_DATE | TIME_SECONDS) + " WARN  " + msg;
      Print(line);
      _WriteFile(line);
   }

   void Error(string msg) {
      string line = TimeToString(TimeCurrent(), TIME_DATE | TIME_SECONDS) + " ERROR " + msg;
      Print(line);
      _WriteFile(line);
   }

   void Critical(string msg) {
      string line = TimeToString(TimeCurrent(), TIME_DATE | TIME_SECONDS) + " CRIT  " + msg;
      Print(line);
      _WriteFile(line);
   }

private:
   void _WriteFile(string line) {
      if(m_fileEnabled && m_handle != INVALID_HANDLE) {
         FileWriteString(m_handle, line + "\n");
         FileFlush(m_handle);
      }
   }
};

#endif
