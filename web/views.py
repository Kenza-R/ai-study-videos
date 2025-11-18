import os
import sys
import threading
import subprocess
from pathlib import Path

from django.conf import settings
from django.contrib.auth import login
from django.contrib.auth.decorators import login_required
from django.contrib.auth.forms import UserCreationForm
from django.http import JsonResponse, HttpResponse, HttpResponseRedirect
from django.shortcuts import render, redirect
from django.urls import reverse

from .forms import PaperUploadForm


def health(request):
    return JsonResponse({"status": "ok"})


def static_debug(request):
    """Debug endpoint to check static files configuration"""
    from django.conf import settings
    from pathlib import Path
    import os
    
    static_root = Path(settings.STATIC_ROOT)
    css_file = static_root / "web" / "css" / "style.css"
    
    info = {
        "STATIC_URL": settings.STATIC_URL,
        "STATIC_ROOT": str(settings.STATIC_ROOT),
        "STATIC_ROOT_exists": static_root.exists(),
        "css_file_path": str(css_file),
        "css_file_exists": css_file.exists(),
        "STATICFILES_STORAGE": settings.STATICFILES_STORAGE,
        "DEBUG": settings.DEBUG,
        "whitenoise_in_middleware": "whitenoise.middleware.WhiteNoiseMiddleware" in settings.MIDDLEWARE,
    }
    
    # Try to read the file if it exists
    if css_file.exists():
        try:
            info["css_file_size"] = css_file.stat().st_size
            info["css_file_readable"] = True
        except Exception as e:
            info["css_file_readable"] = False
            info["css_file_error"] = str(e)
    
    # List files in staticfiles directory
    if static_root.exists():
        try:
            info["staticfiles_contents"] = [str(p.relative_to(static_root)) for p in static_root.rglob("*") if p.is_file()][:20]
        except Exception as e:
            info["staticfiles_list_error"] = str(e)
    
    return JsonResponse(info, indent=2)


def home(request):
    # Render a small landing page with a link to the upload UI
    return render(request, "index.html")


def _start_pipeline_async(pmid: str, output_dir: Path):
    """Start the kyle-code pipeline in a background thread using subprocess.

    This avoids importing the pipeline directly into Django and keeps
    the execution isolated. Output (logs) are written to output_dir/pipeline.log.
    """

    def runner():
        output_dir.mkdir(parents=True, exist_ok=True)
        log_path = output_dir / "pipeline.log"

        # Use the same Python interpreter
        python_exe = sys.executable
        script_path = Path(settings.BASE_DIR) / "kyle-code" / "main.py"

        cmd = [python_exe, str(script_path), "generate-video", pmid, str(output_dir)]

        env = os.environ.copy()

        # Ensure output is written to a log file
        with open(log_path, "ab") as out:
            process = subprocess.Popen(cmd, stdout=out, stderr=out, env=env)
            process.wait()

    thread = threading.Thread(target=runner, daemon=True)
    thread.start()


@login_required
def upload_paper(request):
    """Simple UI to accept a PubMed ID/PMCID and start the pipeline."""
    if request.method == "POST":
        form = PaperUploadForm(request.POST, request.FILES)
        if form.is_valid():
            pmid = form.cleaned_data.get("paper_id")
            # If a file is uploaded we save it and use a folder named by filename
            uploaded = form.cleaned_data.get("file")

            if uploaded:
                # Save uploaded file into media/<basename>/uploaded_file
                name = Path(uploaded.name).stem
                out_dir = Path(settings.MEDIA_ROOT) / name
                out_dir.mkdir(parents=True, exist_ok=True)
                file_path = out_dir / uploaded.name
                with open(file_path, "wb") as f:
                    for chunk in uploaded.chunks():
                        f.write(chunk)

                # TODO: support pipeline from local file; for now, return to status page
                # We'll treat 'name' as an identifier
                pmid = name
            else:
                if not pmid:
                    form.add_error(None, "Provide a PubMed ID or upload a file")
                    return render(request, "upload.html", {"form": form})

                # Normalize pmid
                pmid = pmid.strip()

            # Start pipeline asynchronously and redirect to status page
            output_dir = Path(settings.MEDIA_ROOT) / pmid
            _start_pipeline_async(pmid, output_dir)

            return HttpResponseRedirect(reverse("pipeline_status", args=[pmid]))
    else:
        form = PaperUploadForm()

    return render(request, "upload.html", {"form": form})


def pipeline_status(request, pmid: str):
    """Return a small status page for a running pipeline and a JSON status endpoint."""
    output_dir = Path(settings.MEDIA_ROOT) / pmid
    final_video = output_dir / "final_video.mp4"
    log_path = output_dir / "pipeline.log"

    if request.GET.get("_json"):
        # JSON status endpoint
        status = {
            "pmid": pmid,
            "exists": output_dir.exists(),
            "final_video": final_video.exists(),
            "final_video_url": (
                f"{settings.MEDIA_URL}{pmid}/final_video.mp4" if final_video.exists() else None
            ),
        }
        # include tail of log if present
        if log_path.exists():
            try:
                with open(log_path, "rb") as f:
                    f.seek(max(0, f.tell() - 8192))
                    data = f.read().decode(errors="replace")
            except Exception:
                data = ""
            status["log_tail"] = data

        return JsonResponse(status)

    # Render an HTML status page
    log_tail = ""
    if log_path.exists():
        try:
            with open(log_path, "rb") as f:
                f.seek(max(0, f.tell() - 8192))
                log_tail = f.read().decode(errors="replace")
        except Exception:
            log_tail = "(could not read log)"

    context = {
        "pmid": pmid,
        "final_video_exists": final_video.exists(),
        "final_video_url": f"{settings.MEDIA_URL}{pmid}/final_video.mp4",
        "log_tail": log_tail,
    }

    return render(request, "status.html", context)


def pipeline_result(request, pmid: str):
    output_dir = Path(settings.MEDIA_ROOT) / pmid
    final_video = output_dir / "final_video.mp4"
    if final_video.exists():
        return render(request, "result.html", {"pmid": pmid, "video_url": f"{settings.MEDIA_URL}{pmid}/final_video.mp4"})
    else:
        return HttpResponseRedirect(reverse("pipeline_status", args=[pmid]))


def register(request):
    """User registration view."""
    if request.method == "POST":
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            return redirect("home")
    else:
        form = UserCreationForm()
    return render(request, "registration/register.html", {"form": form})
