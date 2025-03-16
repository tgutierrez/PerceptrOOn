using Microsoft.Extensions.Caching.Memory;
using Perceptr00n.WebUI.App;
using Serilog;
using WebUI.App;

var builder = WebApplication.CreateBuilder(args);

// Add services to the container.
builder.Services.AddRazorPages();

builder.Host.UseSerilog((hostingContext, loggerConfiguration) =>
{
    loggerConfiguration
        .ReadFrom.Configuration(hostingContext.Configuration)
        .MinimumLevel.Debug()
        .Enrich.FromLogContext()
        .WriteTo.Console();
});

builder.Services.AddDistributedMemoryCache();

builder.Services
    .AddSession(options =>
    {
        options.IdleTimeout = TimeSpan.FromMinutes(1);
        options.Cookie.HttpOnly = true;
        options.Cookie.IsEssential = true;
        options.Cookie.Name = "Infer-Session";
    });

builder
    .Services
        .AddMemoryCache(o => o.SizeLimit = 4)
        .AddSingleton<InferenceSessionHandler>();

var app = builder.Build();

// Configure the HTTP request pipeline.
if (!app.Environment.IsDevelopment())
{
    app.UseExceptionHandler("/Error");
    // The default HSTS value is 30 days. You may want to change this for production scenarios, see https://aka.ms/aspnetcore-hsts.
    app.UseHsts();
}

app.UseHttpsRedirection();
app.AddInferenceAPI();
app.UseRouting();
app.UseSession();

app.Use(async (context, next) =>
{
    context.InitializeSessionIfRequired();
    await next();
});

app.UseAuthorization();

app.MapStaticAssets();
app.MapRazorPages()
   .WithStaticAssets();

app.Run();
