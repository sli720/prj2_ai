package cic.cs.unb.ca.ifm;

import cic.cs.unb.ca.flow.FlowMgr;
import cic.cs.unb.ca.jnetpcap.*;
import cic.cs.unb.ca.jnetpcap.worker.FlowGenListener;
import org.jnetpcap.PcapClosedException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import cic.cs.unb.ca.jnetpcap.worker.InsertCsvRow;
import swing.common.SwingUtils;

import java.io.File;
import java.util.*;

import static cic.cs.unb.ca.Sys.FILE_SEP;

public class Cmd {

    public static final Logger logger = LoggerFactory.getLogger(Cmd.class);
    private static final String DividingLine = "-------------------------------------------------------------------------------";
    private static String[] animationChars = new String[]{"|", "/", "-", "\\"};

    public static void main(String[] args) {

        long flowTimeout = 120000000L;
        long activityTimeout = 5000000L;
        String rootPath = System.getProperty("user.dir");
        String pcapPath;
        String outPath;

        /* Select path for reading all .pcap files */
        /*if(args.length<1 || args[0]==null) {
            pcapPath = rootPath+"/data/in/";
        }else {
        }*/

        /* Select path for writing all .csv files */
        /*if(args.length<2 || args[1]==null) {
            outPath = rootPath+"/data/out/";
        }else {
        }*/

        if (args.length < 1) {
            logger.info("Please select input folder!");
            return;
        }
        pcapPath = args[0];
        File in = new File(pcapPath);

        if(!in.isDirectory()){
            logger.info("The input folder does not exist! -> {}",pcapPath);
            return;
        }

        if (args.length < 2) {
            logger.info("Please select output folder!");
            return;
        }
        outPath = args[1];
        File out = new File(outPath);
        if (out.isFile()) {
            logger.info("The out folder does not exist! -> {}",outPath);
            return;
        }

        if (args.length < 3) {
            logger.info("Please supply the attacks string (labels separated by ':')!");
            return;
        }
        Label attackTypes[] = Label.getAttackTypeLabels(args[2]);
        logger.info("You select: {}",pcapPath);
        logger.info("Out folder: {}",outPath);
        logger.info("Attack Types: {}", Arrays.toString(attackTypes));

        readPcapFiles(in, attackTypes, outPath,flowTimeout,activityTimeout);
}

    private static void readPcapFiles(File inDir, Label[] attackTypes, String outPath, long flowTimeout, long activityTimeout) {
        if(inDir==null || attackTypes==null || outPath==null ) {
            return;
        }
        String fileName = "features";

        if(!outPath.endsWith(FILE_SEP)){
            outPath += FILE_SEP;
        }

        File saveFileFullPath = new File(outPath+fileName+FlowMgr.FLOW_SUFFIX);

        if (saveFileFullPath.exists()) {
           if (!saveFileFullPath.delete()) {
               System.out.println("Save file can not be deleted");
           }
        }

        FlowGenerator flowGen = new FlowGenerator(true, flowTimeout, activityTimeout);
        flowGen.addFlowListener(new FlowListener(fileName,outPath));
        boolean readIP6 = false;
        boolean readIP4 = true;
        Map<PacketReader, Label> packetReaderLabels = new HashMap<>();
        Map<PacketReader, BasicPacketInfo> nextPackets = new HashMap<>();
        File[] pcapFiles = inDir.listFiles(file -> file.isFile() && file.getName().endsWith(".pcap"));
        if(pcapFiles != null) {
            for (File pcapFile : pcapFiles) {
                // Use BENIGN as default, if no attack type label matches
                Label label = Label.BENIGN;
                for (Label attackType : attackTypes) {
                    if(pcapFile.getName().toLowerCase().startsWith(attackType.toString().toLowerCase())) {
                        label = attackType;
                        break;
                    }
                }
                PacketReader packetReader = new PacketReader(pcapFile.getPath(), readIP4, readIP6);
                packetReaderLabels.put(packetReader, label);
                logger.info("Opened file " + pcapFile + " (Label: " + label + ')');
                try {
                    nextPackets.put(packetReader, packetReader.nextPacket());
                } catch (PcapClosedException e) {
                    // Pcap file seems to be empty, so we can ignore it
                }
            }
        }

        int nValid=0;
        int nTotal=0;
        int nDiscarded = 0;
        long start = System.currentTimeMillis();
        int i=0;
        // Continue to read until all pcap files have been read completely
        while(!nextPackets.isEmpty()) {
            /*i = (i)%animationChars.length;
            System.out.print("Working on "+ inputFile+" "+ animationChars[i] +"\r");*/

            // First find out which pcap file contains the next packet according to timestamp
            PacketReader nextPacketReader = null;
            long nextTimestamp = Long.MAX_VALUE;
            for(Map.Entry<PacketReader, BasicPacketInfo> nextPacket : nextPackets.entrySet()) {
                if(nextPacket.getValue().getTimeStamp() <= nextTimestamp) {
                    nextTimestamp = nextPacket.getValue().getTimeStamp();
                    nextPacketReader = nextPacket.getKey();
                }
            }

            // Now add that packet to the corresponding flow (or create a new flow if required)
            flowGen.addPacket(packetReaderLabels.get(nextPacketReader), nextPackets.get(nextPacketReader));

            // Then read the new "next" packet from that pcap file
            BasicPacketInfo nextPacket = null;
            // Continue to read until the next packet is a valid packet (i.e. not null)
            while(nextPacket == null) {
                try {
                    nextPacket = nextPacketReader.nextPacket();
                }catch(PcapClosedException e) {
                    // We read all packets of that pcap file
                    nextPackets.remove(nextPacketReader);
                    break;
                }
                nTotal++;
                if (nextPacket != null) {
                    // If we found a valid packet, put it into the next packet map
                    nextPackets.put(nextPacketReader, nextPacket);
                    nValid++;
                } else {
                    nDiscarded++;
                }
            }
            i++;
        }

        flowGen.dumpLabeledCurrentFlow(saveFileFullPath.getPath(), FlowFeature.getHeader());

        long lines = SwingUtils.countLines(saveFileFullPath.getPath());

        System.out.println(String.format("%s is done. total %d flows ",fileName,lines));
        System.out.println(String.format("Packet stats: Total=%d,Valid=%d,Discarded=%d",nTotal,nValid,nDiscarded));
        System.out.println(DividingLine);

        //long end = System.currentTimeMillis();
        //logger.info(String.format("Done! in %d seconds",((end-start)/1000)));
        //logger.info(String.format("\t Total packets: %d",nTotal));
        //logger.info(String.format("\t Valid packets: %d",nValid));
        //logger.info(String.format("\t Ignored packets:%d %d ", nDiscarded,(nTotal-nValid)));
        //logger.info(String.format("PCAP duration %d seconds",((packetReader.getLastPacket()- packetReader.getFirstPacket())/1000)));
        //int singleTotal = flowGen.dumpLabeledFlowBasedFeatures(outPath, fileName+ FlowMgr.FLOW_SUFFIX, FlowFeature.getHeader());
        //logger.info(String.format("Number of Flows: %d",singleTotal));
        //logger.info("{} is done,Total {} flows",inputFile,singleTotal);
        //System.out.println(String.format("%s is done,Total %d flows", inputFile, singleTotal));
    }

    static class FlowListener implements FlowGenListener {

        private String fileName;

        private String outPath;

        private long cnt;

        public FlowListener(String fileName, String outPath) {
            this.fileName = fileName;
            this.outPath = outPath;
        }

        @Override
        public void onFlowGenerated(BasicFlow flow) {

            String flowDump = flow.dumpFlowBasedFeaturesEx();
            List<String> flowStringList = new ArrayList<>();
            flowStringList.add(flowDump);
            InsertCsvRow.insert(FlowFeature.getHeader(),flowStringList,outPath,fileName+ FlowMgr.FLOW_SUFFIX);

            cnt++;

            String console = String.format("%s -> %d flows \r", fileName,cnt);

            System.out.print(console);
        }
    }

}
