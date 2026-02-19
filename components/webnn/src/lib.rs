//! Minimal WebNN manager component (plumbing only).
//!
//! This crate only provides a `WebNNMsg` message type and a tiny
//! manager factory used to obtain a `GenericSender<WebNNMsg>` that
//! can be passed through the Constellation → ScriptThread → GlobalScope
//! plumbing. The manager is a stub (no backend logic yet).

use std::thread;

use base::generic_channel::{GenericReceiver, GenericSender, channel};
use log::debug;
use webnn_traits::WebNNMsg;

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
    loop {
        match receiver.recv() {
            Ok(msg) => match msg {
                WebNNMsg::Exit => {
                    debug!("webnn manager exiting");
                    break;
                },
            },
            Err(_) => break,
        }
    }
    debug!("webnn manager stopped");
}
