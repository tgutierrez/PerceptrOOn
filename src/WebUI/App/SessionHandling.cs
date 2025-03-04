using Microsoft.Extensions.Caching.Memory;
using System.Xml.Serialization;
using WebUI.App;

namespace Perceptr00n.WebUI.App
{
    public static class SessionHandling
    {
        private static string SessionKey = "SessionKey";

        public static void InitializeSessionIfRequired(this HttpContext context)
        {
            if (context == null || context.Session == null) return;

            if (!string.IsNullOrEmpty(context.Session.GetString(SessionKey))) return;

            context.Session.StartSession(context);
        }

        public static void StartSession(this ISession session, HttpContext context)
        {
            var val = session.GetString(SessionKey);
            if (!string.IsNullOrEmpty(val)) return;

            var sessionId = Guid.NewGuid();

            session.SetString(SessionKey, sessionId.ToString());

            var inferenceSession = context.RequestServices.GetRequiredService<InferenceSessionHandler>();
            inferenceSession.CreateSession(sessionId);
        }

        public static InferenceHandler GetInferenceSession(this HttpContext context)
        {
            var inferenceSession = context.RequestServices.GetRequiredService<InferenceSessionHandler>();
            var sessionId = context.Session.GetString(SessionKey);
            return inferenceSession.GetSession(Guid.Parse(sessionId!));
        }
    }

    public class InferenceSessionHandler 
    {
        private readonly IMemoryCache _memoryCache;
        public InferenceSessionHandler(IMemoryCache memoryCache)
        {
            _memoryCache = memoryCache;
        }

        public void CreateSession(Guid id)
        {
            _memoryCache.Set(id, new InferenceHandler(id), new MemoryCacheEntryOptions
            {
                Size = 1,
                SlidingExpiration = TimeSpan.FromMinutes(5)
            });
        }

        public InferenceHandler GetSession(Guid id)
        {
            return _memoryCache.Get<InferenceHandler>(id)!;
        }


    }
}
