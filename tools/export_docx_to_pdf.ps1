param(
    [Parameter(Mandatory=$true)]
    [string]$DocxPath,
    [Parameter(Mandatory=$true)]
    [string]$PdfPath
)

$ErrorActionPreference = "Stop"

$word = $null
$doc = $null

try {
    $word = New-Object -ComObject Word.Application
    $word.Visible = $false
    $word.DisplayAlerts = 0
    $doc = $word.Documents.Open($DocxPath)
    $doc.ExportAsFixedFormat($PdfPath, 17)
}
finally {
    if ($null -ne $doc) {
        $doc.Close($false)
    }
    if ($null -ne $word) {
        $word.Quit()
    }
}

Write-Host "Exported PDF: $PdfPath"
