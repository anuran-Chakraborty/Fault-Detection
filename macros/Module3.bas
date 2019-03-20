Attribute VB_Name = "Module2"
Sub delete_row_csv()
'
' delte_row Macro
'

'
    Dim WS As Excel.Worksheet
    Dim SaveToDirectory As String
    
    SaveToDirectory = "F:\Fault Detection\csv_files\"

    For Each WS In ThisWorkbook.Worksheets
        WS.Range("A1").EntireRow.Delete
    Next
End Sub
Sub insertrow()
    Dim WS As Excel.Worksheet
    Dim SaveToDirectory As String
    
    SaveToDirectory = "F:\Fault Detection\csv_files\"

    For Each WS In ThisWorkbook.Worksheets
        WS.Range("A1").EntireRow.Insert
    Next
End Sub
Sub genvalues()
Attribute genvalues.VB_ProcData.VB_Invoke_Func = " \n14"
'
' genvalues Macro
'

'
    Dim WS As Excel.Worksheet
    Dim SaveToDirectory As String
    Dim formulae As String
    SaveToDirectory = "F:\Fault Detection\csv_files\"
    
    For Each WS In ThisWorkbook.Worksheets
        WS.Cells(1, 1).FormulaR1C1 = "c01"
        WS.Range("A1").AutoFill Destination:=WS.Range("A1:Y1"), Type:=xlFillDefault
    Next
End Sub
Sub save_as_csv()
    Dim WS As Excel.Worksheet
    Dim SaveToDirectory As String

    SaveToDirectory = "F:\Fault Detection\csv_files\"

    For Each WS In ThisWorkbook.Worksheets
        WS.SaveAs SaveToDirectory & WS.Name, xlCSV
    Next

End Sub
