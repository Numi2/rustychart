#[cfg(feature = "ssr")]
mod server {
    use axum::Router;
    use leptos::*;
    use leptos_axum::{generate_route_list, LeptosRoutes};
    use ui::App;

    pub async fn run() {
        // Load Leptos config (defaults are fine for now)
        let conf = get_configuration(None).await.expect("load config");
        let leptos_options = conf.leptos_options;
        let addr = leptos_options.site_addr;
        let routes = generate_route_list(App);

        let app = Router::new()
            .leptos_routes(&leptos_options, routes, App)
            .with_state(leptos_options);

        println!("UI server listening on http://{}", addr);
        axum::serve(
            tokio::net::TcpListener::bind(addr).await.expect("bind"),
            app.into_make_service(),
        )
        .await
        .unwrap();
    }
}

#[cfg(feature = "ssr")]
#[tokio::main]
async fn main() {
    server::run().await;
}

#[cfg(not(feature = "ssr"))]
fn main() {}
