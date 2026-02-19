use std::collections::HashMap;
use std::thread;

use base::generic_channel::{GenericReceiver, GenericSender, channel};
use log::debug;
use webnn_traits::{ContextId, WebNNMsg};

#[derive(Debug)]
struct ContextInfo {
    // Placeholder for future backend-specific context state.
}

/// Create a new WebNN manager and return the `GenericSender<WebNNMsg>`
/// together with the `JoinHandle` for the manager thread.
///
/// Returning the join handle allows the caller (the `Constellation`) to
/// join the manager thread during shutdown.
pub fn new_webnn_manager() -> (GenericSender<WebNNMsg>, std::thread::JoinHandle<()>) {
    let (tx, rx): (GenericSender<WebNNMsg>, GenericReceiver<WebNNMsg>) =
        channel().expect("webnn channel");

    let handle = thread::Builder::new()
        .name("WebNNManager".into())
        .spawn(move || run_manager(rx))
        .expect("failed to spawn WebNN manager thread");

    (tx, handle)
}

fn run_manager(receiver: GenericReceiver<WebNNMsg>) {
    debug!("webnn manager started");

    let mut contexts: HashMap<ContextId, ContextInfo> = HashMap::new();

    loop {
        match receiver.recv() {
            Ok(msg) => match msg {
                WebNNMsg::Exit => {
                    debug!("webnn manager exiting");
                    break;
                },
                WebNNMsg::NewContext(id) => {
                    debug!("webnn manager: NewContext {:?}", id);
                    contexts.insert(id, ContextInfo {});
                },
                WebNNMsg::DestroyContext(id) => {
                    debug!("webnn manager: DestroyContext {:?}", id);
                    contexts.remove(&id);
                },
            },
            Err(_) => break,
        }
    }
    debug!("webnn manager stopped");
}
