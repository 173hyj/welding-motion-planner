using DLAT.JTReader;
using System.Text.Json;
var source = new JTFile(args[0]);
// JT metadata is lazy; materialize properties before reading the TxKin buffer.
var metadata = source.segments.Values.SelectMany(s => s.elements ?? new List<Element>())
.Where(e => e.elementData is PropertyProxyMetaData)
.Select(e => new {e.objectID, properties=((PropertyProxyMetaData)e.elementData).property.ToDictionary(p => p.Key,p => p.Value.Item2?.ToString())}).ToArray();
File.WriteAllText(args[1],JsonSerializer.Serialize(new {source=args[0],version=source.Version,kinematicsSource=source.txKinModelingBuffer,metadata},new JsonSerializerOptions{WriteIndented=true}));
Console.WriteLine(source.txKinModelingBuffer ?? "No kinematics buffer");
