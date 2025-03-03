using System.Xml.Serialization;

namespace Perceptr00n.WebUI.App
{
    public static class SessionHandling
    {
        private static string SessionKey = "SessionKey";

        public static void InitializeSessionIfRequired(this HttpContext context)
        {
            if (context == null || context.Session == null) return;

            if (!string.IsNullOrEmpty(context.Session.GetString(SessionKey))) return;

            context.Session.StartSession();
        }

        public static void StartSession(this ISession session)
        {
            var val = session.GetString(SessionKey);
            if (!string.IsNullOrEmpty(val)) return;


        }
    }
}
