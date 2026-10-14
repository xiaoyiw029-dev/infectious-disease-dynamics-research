param(
    [Parameter(Mandatory=$true)]
    [string]$DocxPath
)

$ErrorActionPreference = "Stop"

$lambda = [string][char]0x03BB
$sigma = [string][char]0x03C3
$gamma = [string][char]0x03B3
$beta = [string][char]0x03B2
$mu = [string][char]0x03BC
$Lambda = [string][char]0x039B
$theta = [string][char]0x03B8
$yhat = [string][char]0x0177
$integral = [string][char]0x222B
$sqrt = [string][char]0x221A

$equations = [ordered]@{}
$equations.Add("[[EQ_FORCE]]", $lambda + "_h=" + $beta + "_h I_v/N_v,   " + $lambda + "_v=" + $beta + "_v I_h/N_h")
$equations.Add("[[EQ_M1]]", "dS_h/dt=-" + $lambda + "_h S_h")
$equations.Add("[[EQ_M2]]", "dE_h/dt=" + $lambda + "_h S_h-" + $sigma + "_h E_h")
$equations.Add("[[EQ_M3]]", "dI_h/dt=" + $sigma + "_h E_h-" + $gamma + "_h I_h")
$equations.Add("[[EQ_M4]]", "dR_h/dt=" + $gamma + "_h I_h")
$equations.Add("[[EQ_M5]]", "dS_v/dt=" + $Lambda + "_v-" + $lambda + "_v S_v-" + $mu + "_v S_v")
$equations.Add("[[EQ_M6]]", "dE_v/dt=" + $lambda + "_v S_v-(" + $sigma + "_v+" + $mu + "_v)E_v")
$equations.Add("[[EQ_M7]]", "dI_v/dt=" + $sigma + "_v E_v-" + $mu + "_v I_v")
$equations.Add("[[EQ_DFE]]", "E_0=(N_h,0,0,0," + $Lambda + "_v/" + $mu + "_v,0,0)")
$equations.Add("[[EQ_R0]]", "R_0=" + $sqrt + "((" + $beta + "_h " + $beta + "_v " + $sigma + "_v)/(" + $gamma + "_h " + $mu + "_v(" + $sigma + "_v+" + $mu + "_v)))")
$equations.Add("[[EQ_OBS]]", $yhat + "_i(" + $theta + ")=q" + $integral + "_(7(i-1))^(7i) " + $sigma + "_h E_h(t;" + $theta + ") dt")

$word = $null
$doc = $null

try {
    $word = New-Object -ComObject Word.Application
    $word.Visible = $false
    $word.DisplayAlerts = 0
    $doc = $word.Documents.Open($DocxPath)

    foreach ($item in $equations.GetEnumerator()) {
        $range = $doc.Content
        $find = $range.Find
        $find.ClearFormatting()
        $find.Text = $item.Key
        $found = $find.Execute()
        if (-not $found) {
            throw "Placeholder not found: $($item.Key)"
        }

        $range.Text = $item.Value
        $range.Font.Name = "Cambria Math"
        $range.Font.Size = 10.5
        $range.ParagraphFormat.Alignment = 1

        $null = $doc.OMaths.Add($range)
        $range.OMaths.Item(1).BuildUp()
    }

    $doc.Save()
}
finally {
    if ($null -ne $doc) {
        $doc.Close($false)
    }
    if ($null -ne $word) {
        $word.Quit()
    }
}

Write-Host "Converted placeholders to native Word equations: $DocxPath"
