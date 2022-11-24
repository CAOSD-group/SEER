from PyPDF2 import PdfFileWriter, PdfFileReader

inputpdf = PdfFileReader(open("salidaGTvsEstimated0.pdf", "rb"))

algo = ["x264", "lrzip", "Dune", "LLVM", "BerkeleyDBC",
        "Hipacc", "7z", "JavaGC", "Polly", "VP9"]
for i in range(inputpdf.numPages):
    output = PdfFileWriter()
    output.addPage(inputpdf.getPage(i))
    with open("%s.pdf" % algo[i], "wb") as outputStream:
        output.write(outputStream)