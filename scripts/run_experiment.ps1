$ErrorActionPreference = "Stop"

if ($args.Count -gt 0) {
    $RUNS = $args[0]
} else {
    $RUNS = 30
}

Write-Host ">>> Krok 1: Budowanie obrazu Docker 'cma_experiments'..."
docker build -t cma_experiments .

Write-Host ""
Write-Host ">>> Krok 2: Uruchamianie eksperymentu ($RUNS przebiegow)..."
docker run `
    --rm `
    -it `
    -v "${PWD}/data:/app/data" `
    -v "${PWD}/plots:/app/plots" `
    cma_experiments $RUNS

Write-Host ""
Write-Host ">>> Skrypt zakonczył dzialanie. Wyniki w folderach 'data' i 'plots'."
