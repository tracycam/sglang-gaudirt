"""SGLang entry point for the GaudiRT device."""

_compiler_registered = False


def activate():
    global _compiler_registered
    import gaudirt  # Registers the PrivateUse1 device and audits old-bridge isolation.

    if not _compiler_registered:
        import torch._dynamo
        from gaudirt.compiler import gaudirt_backend

        torch._dynamo.register_backend(gaudirt_backend, name="gaudirt")
        _compiler_registered = True
    return "sglang_gaudirt.platform.GaudiRTSRTPlatform"
