# second.py
from functools import lru_cache
import os, subprocess

from fastapi import FastAPI
import imagej, jpype
from scyjava import config, jimport

api = FastAPI()

def _detect_java_home() -> str | None:
    jh = os.environ.get("JAVA_HOME")
    if jh and os.path.isdir(jh):
        return jh
    # macOS: prefer JDK 17 then 11
    for v in ("17", "11"):
        try:
            out = subprocess.check_output(["/usr/libexec/java_home", "-v", v], text=True).strip()
            if out and os.path.isdir(out):
                return out
        except Exception:
            pass
    return None

@lru_cache(maxsize=1)
def get_ij():
    # Make sure a JDK is visible via env (no scyjava.set_java_home!)
    print("Initializing ImageJ...")
    jh = _detect_java_home()
    print("Using JAVA_HOME:", jh)
    if not jh:
        raise RuntimeError(
            "No JDK found. Install OpenJDK and set JAVA_HOME, e.g.:\n"
            "  brew install openjdk@17\n"
            "  sudo mkdir -p /Library/Java/JavaVirtualMachines && "
            "  sudo ln -sfn /opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk "
            "               /Library/Java/JavaVirtualMachines/openjdk-17.jdk\n"
            "  export JAVA_HOME=$(/usr/libexec/java_home -v 17)\n"
        )
    os.environ["JAVA_HOME"] = jh
    # Avoid Java auto-fetch path that was throwing earlier
    os.environ.setdefault("SCYJAVA_FETCH_JAVA", "never")
    # JVM options must be set before imagej.init()
    config.add_option("-Xmx2g")
    config.add_option("-Djava.awt.headless=true")

    print("Starting JVM...")
    ij = imagej.init("net.imagej:imagej:2.14.0", mode="headless")
    print(f"ImageJ version: {ij.getVersion()}")

    print()
    print("JVM started.")
    if not jpype.isThreadAttachedToJVM():
        jpype.attachThreadToJVM()
    return ij

@api.get("/ping")
def ping():
    ij = get_ij()
    print('GGG')
    print(f"ImageJ version: {ij.getVersion()}")


    System = jimport("java.lang.System")


    print(f"ImageJ version: {ij.getVersion()}")

    return {
        "imagej_version": {ij.getVersion()},
        "java_version": System.getProperty("java.version"),
        "java_vendor": System.getProperty("java.vendor"),
        "java_home": System.getProperty("java.home"),
    }
