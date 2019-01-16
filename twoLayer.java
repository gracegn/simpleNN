import java.util.ArrayList;
import java.util.Scanner;
import java.util.*;
/**
 * two layer neural network aka 2 groups of weights with i inputs
 * X  = 1xi (input layer)
 * W1 = ixh
 * H  = 1xh (hidden layer)
 * W2 = hxy
 * Y  = 1xy (output layer)
 * note that h, the number of hidden neurons, will be determined by taking the
 * rounded down mean of i and y as reasoned in the following
 * https://stats.stackexchange.com/questions/181/how-to-choose-the-number-of-hidden-layers-and-nodes-in-a-feedforward-neural-netw
 * 
 * equations:
 * H = f(XW1 + B1)
 * Y = f(XW2 + B2)
 * J = error = (1/2)(Y-Y*)^2 where Y = target or ideal, Y* = output or model
 * 
 * net = matrix product of input & weights + bias  ie i_1*w_1 + i_2*w_2 + b1
 *   net' = coefficient of the weight
 * out = activation function applied to the net  ie 1/(1 + e^(-x))
 *   out' = 
 * Etot = sum of the errors
 *   Etot' = 
 * 
 */
public class twoLayer
{
    public static void main()
    {
        Scanner scInt = new Scanner(System.in);
        Scanner scD = new Scanner(System.in);
        
        //get inputs & target outputs
        System.out.println("Enter the # of inputs and the inputs themselves.");
        int ii = scInt.nextInt();
        ArrayList<Double> inp = new ArrayList<Double>();
        for(int i = 0; i < ii; i++)
            inp.add(scD.nextDouble());
        
        System.out.println("Enter the # of target outputs and the actual outputs.");
        int yy = scInt.nextInt();
        ArrayList<Double> target = new ArrayList<Double>();
        for(int i = 0; i < yy; i++)
            target.add(scD.nextDouble());
        
        
        
        //*************************************************************************
        //get # of hidden neurons, initial guesses for the 2 weight layers, biases
        int hh = (int) Math.floor((ii+yy)/2);
        final double INIT = 0.15;
        final double INCRE = 0.05;
        //weight1 matrix = i x h, so we'll have h elements in the arraylist, 
        //and each element is an arraylist of i weights FOR WEIGHT1 GUESS
        List<List<Double>> w1guess = new ArrayList<List<Double>>();
        double b1 = loadInitial(hh, ii, INIT, w1guess);
        //weight2 matrix = h x y, so y elements and each is an arr of h weights
        List<List<Double>> w2guess = new ArrayList<List<Double>>();
        double b2 = loadInitial(yy, hh, INCRE*(1+ii*hh)+INIT, w2guess);
        
        /*System.out.println("w1guess = " + w1guess);
        System.out.println("b1 = " + b1);
        System.out.println("w2guess = " + w2guess);
        System.out.println("b2 = " + b2);*/
        
        
        
        //*************************************************************************
        //now we begin FORWARDS 
        /*
         * we have inputs, weight1 guesses, weight2 guesses, target outputs
         * USING FIRST WEIGHTS
         * inputs -> net h1 with w_i*i_i + b1
         * net h1 -> out h1 with actFunc(net h1)
         * 
         * USING SECOND WEIGHTS
         * o_i -> net o1 with w_i*o_i + b2
         * net o1 -> out o1 with actFunc(net o1)
         */
        ArrayList<Double> neth = new ArrayList<Double>();
        net(b1, inp, w1guess, neth);
        
        ArrayList<Double> outh = new ArrayList<Double>();
        outActFunc(neth, outh);
        
        ArrayList<Double> neto = new ArrayList<Double>();
        net(b2, outh, w2guess, neto);
        
        ArrayList<Double> outo = new ArrayList<Double>();
        outActFunc(neto, outo);
        
        /*System.out.println("neth = " + neth);
        System.out.println("outh = " + outh);
        System.out.println("neto = " + neto);
        System.out.println("outo = " + outo);*/
        
        
        
        //*************************************************************************
        //now we begin BACKWARDS
        /*
         * get Etotal = sum of (1/2)(target-out)^2
         * USE PARTIAL DERIVATIVES TO UPDATE SECOND/OUTER SET OF WEIGHTS
         * apply delta rule to get derivative of Etot in terms of each weight
         * new weight = original - learning rate * derivative
         * 
         * repeat process of derivatives etc to update first/inner set of weights
         * 
         * 
         * delta rule for second/outer:
         * d(Etot)/d(w_5) = d(Etot)/d(outo1) *
         *                  d(outo1)/d(neto1) *
         *                  d(neto1)/d(w_5)
         * = (outo1 - targeto1) * outo1(1-outo1) * outh1
         * 
         * 
         * 
         * delta rule for first/inner:
         * d(Etot)/d(w_1) = d(Etot)/d(outh1) *
         *                  d(outh1)/d(neth1) *
         *                  d(neth1)/d(w_1)
         * FIRST MULTIPLIEE: d(Etot)/d(outh1)
         *  = d(Etot)/d(outo1) * d(outo1)/d(neto1) * d(neto1)/d(outh1)
         *  = (outo1 - targeto1) * outo1(1-outo1) * w5
         * SECOND MULTIPLIEE: d(outh1)/d(neth1)
         *  = 
         * 
         */
        
        double Etot = totalError(target, outo);
        final double LEARNRATE = 0.5;
        double derEtotW;    //for temp holding the der value
        //note that neto1 = w_5*outh1 + w_6*outh2 + b2
        //note that neth1 = w_1*i_1 + w_2*i_2 + b1
        
        //d(Etot)/d(w_5) = d(Etot)/d(outo1) * d(outo1)/d(neto1) * d(neto1)/d(w_5)
        ArrayList<Double> newW2 = new ArrayList<Double>();
        for(int i = 0; i < w2guess.size(); i++)
        {
            for(int j = 0; j < w2guess.get(0).size(); j++)
            {
                derEtotW = eOuto(i, target, outo) *  //FIRST MULTIPLIEE
                           outNet(i, outo) *         //SECOND MULTIPLIEE
                           netWi(j, outh);          //THIRD MULTIPLIEE
                newW2.add(w2guess.get(i).get(j) - LEARNRATE*derEtotW);
            }
        }
        
        
        
        ArrayList<Double> newW1 = new ArrayList<Double>();
        for(int i = 0; i < w1guess.size(); i++)
        {
            for(int j = 0; j < w1guess.get(0).size(); j++)
            {
                //the first one is special bc we have to break it up into the 
                //different things
                double first = eOuto(0, target, outo)*outNet(0, outo)*
                               netoOuth(i, w2guess.get(0));
                double second = eOuto(1, target, outo)*outNet(1, outo)*
                                netoOuth(i, w2guess.get(1));
                double EtotOuth =  first + second;       //FIRST MULTIPLIEE
                
                double s = outNet(i, outh);              //SECOND MULTIPLIEE
                double t = netWi(j, inp);                //THIRD MULTIPLIEE
                derEtotW = EtotOuth*s*t;
                newW1.add(w1guess.get(i).get(j) - LEARNRATE*derEtotW);
            }
        }
        System.out.println("newW2 = " + newW2);
        System.out.println("newW1 = " + newW1);
    }
    
    
    
    
    
    
    
    //*****************************************************************************
    //                                  FORWARDS
    //*****************************************************************************
    
    public static void net(double bias, ArrayList<Double> inp, List<List<Double>> w,
                           ArrayList<Double> h)
    {
        //we cycle through the elements of w and get matrix product by getting ie
        //[i_1*w_1 + i_2*w_2 + bias, i_1*w_3 + i_2*w_4 + bias]
        //note that inp.size() = w.get(0).size()
        for(int i = 0; i < w.size(); i++)
        {
            double subsum = bias;
            for(int j = 0; j < w.get(0).size(); j++)
                subsum += inp.get(j) * w.get(i).get(j);
            h.add(subsum);
        }
    }
    
    public static void outActFunc(ArrayList<Double> h, ArrayList<Double> out)
    {
        for(int i = 0; i < h.size(); i++)
            out.add(actFunc(h.get(i)));
    }
    
    public static double actFunc(double x)
    {
        return 1 / (1 + Math.pow(Math.E, -1*x));
    }
    
    public static double loadInitial(int outer, int inner, double temp, List<List<Double>> arr)
    {
        temp = Math.floor(temp * 100) / 100;
        for(int i = 0; i < outer; i++)
        {
            arr.add(new ArrayList<Double>());
            for(int j = 0; j < inner; j++)
            {
                arr.get(i).add(temp);
                temp += + 0.05;
                temp = Math.floor(temp * 100) / 100;
            }
        }
        return temp;
    }
    
    
    
    
    
    //*****************************************************************************
    //                                  BACKWARDS
    //*****************************************************************************
    
    public static double totalError(ArrayList<Double> target, ArrayList<Double> out)
    {
        double Etot = 0;
        for(int i = 0; i < target.size(); i++)
        {
            Etot += Math.pow(target.get(i) - out.get(i), 2) / 2;
        }
        return Etot;
    }
    
    //THE SECOND/OUTER LAYER
    
    public static double eOuto(int THING, ArrayList<Double> target, ArrayList<Double> out)
    {
        return out.get(THING) - target.get(THING);
    }
    
    public static double outNet(int THING, ArrayList<Double> out)
    {
        return out.get(THING) * (1 - out.get(THING));
    }
    
    public static double netWi(int THING2, ArrayList<Double> inp)
    {
        return inp.get(THING2);
    }
    
    //THE FIRST/INNER LAYER
    
    public static double netoOuth(int THING, List<Double> inp)
    {
        return inp.get(THING);
    }
}
