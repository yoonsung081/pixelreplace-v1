const params = new URLSearchParams(self.location.search)
const scriptName = params.get("script") || "./obamify.js"

try {
  const obamifyModule = await import(scriptName)
  const wasmName = scriptName.replace(".js", "_bg.wasm")

  // Use initSync or fetch-based init to avoid deprecated string-path overload.
  // Prefer the module { module_or_path } object form which is the current API.
  const wasmResponse = await fetch(wasmName)
  const wasmBytes = await wasmResponse.arrayBuffer()
  await obamifyModule.default({ module_or_path: wasmBytes })

  // Kick off the worker event loop
  if (typeof obamifyModule.worker_entry === "function") {
    obamifyModule.worker_entry()
  }
} catch (e) {
  console.error("worker failed to initialize:", e)
  throw e
}
