import logging
import pytrap

ALF_FORM = "ipaddr DST_IP,ipaddr SRC_IP,uint16 SRC_PORT,uint16 DST_PORT,uint64 BYTES,uint64 BYTES_REV,uint32 PACKETS,uint32 PACKETS_REV,uint8 PREDICTED_CLASS,float PREDICTED_UNCERT"

class Predictor:
    """
    Implements predictor.
    1. Read from Nemea input interface.
    2. Predict. If no model is available, just send it through.
    3. It keeps the UniRec template and adds 2 fields:
        a) Prediction of class.
        b) Measure of uncertainty of class.
    4. Pass the flow.
    """
    def run(self, ifc_spec, buffer_size = 100000) -> None:
        ctx = pytrap.TrapCtx()
        ctx.init(["-i", ifc_spec], 1, 1)
        ctx.setRequiredFmt(0)
        ctx.setDataFmt(0, pytrap.FMT_UNIREC, ALF_FORM)
        rec = None
        specIn = ""
        out = pytrap.UnirecTemplate(ALF_FORM)
        out.createMessage(8192)
        buffer = []
        loop = True
        while loop:
            try:
                data = ctx.recv()
            except pytrap.FormatChanged as e:
                _, specIn = ctx.getDataFmt(0)
                rec = pytrap.UnirecTemplate(specIn)
                data = e.data
            if len(data) <= 1:
                loop = False
            recvFlow = rec.copy()
            recvFlow.setData(data)
            buffer.append(recvFlow)
            if len(buffer) >= buffer_size:
                self.process(ctx,buffer,out)
        if len(buffer) > 0:
            self.process(ctx, buffer,out)
        ctx.finalize()

    def process(self, ctx, buffer,out):
        for flow in buffer:
            out.DST_IP = flow.DST_IP
            out.SRC_IP = flow.SRC_IP
            out.DST_PORT = flow.DST_PORT
            out.SRC_PORT = flow.SRC_PORT
            out.BYTES = flow.BYTES
            out.BYTES_REV = flow.BYTES_REV
            out.PACKETS = flow.PACKETS
            out.PACKETS_REV = flow.PACKETS_REV
            out.PREDICTED_CLASS = 1
            out.PREDICTED_UNCERT = 0.56
            ctx.send(out.getData(), 0)


    def predict(self, ctx, buffer) -> tuple[int, float]:
        pass

    def send_prediction(self, ctx, data):
        pass

    

        

if __name__ == "__main__": 
    p = Predictor()
    p.run("u:alf_predictor_socket,u:sink")
