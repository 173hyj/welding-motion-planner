using DLAT.JTReader;
using System.Text.Json;

if (args.Length < 2) throw new ArgumentException("Usage: converter source.jt destination.json");
var source = new JTFile(args[0]);
var rawPmi = new List<object>();
foreach(var segment in source.segments.Values) foreach(var element in segment.elements ?? new List<Element>()) {
    if (element.elementType == typeof(PMIManagerMetaData) && element.dataStream != null) {
        var stream = element.dataStream;
        rawPmi.Add(new { id = element.objectID, segment = segment.segmentID.ToString(), bytes = Convert.ToBase64String(stream.ReadBytes(0L, (int)stream.Length)) });
        stream.Position = 0;
    }
}
var lsg = new LSG(source);
foreach (var element in source.elements.Values.ToList()) {
    if (element.elementData is BaseNodeData && !lsg.nodes.ContainsKey(element.objectID))
        _ = new LSGNode(lsg, element);
}
var meshes = new Dictionary<string, object>();
var nodes = new List<object>();
foreach (var node in lsg.nodes.Values) {
    var attributes = node.attributes.Select(e => e.elementData).ToList();
    var transform = attributes.OfType<GeometricTransformAttributeData>().FirstOrDefault()?.matrix.raw;
    var material = attributes.OfType<MaterialAttributeData>().FirstOrDefault()?.diffuseColor;
    var ids = new List<string>();
    foreach (var entry in node.property) {
        if (entry.Value is not DataSegment segment) continue;
        var shape = segment.elements.Select(e => e.elementData).OfType<Tri_StripSetShapeLODData>().FirstOrDefault();
        if (shape == null) continue;
        var id = segment.segmentID.ToString();
        if (!meshes.ContainsKey(id)) {
            var vertices = shape.GetVertices();
            var faces = shape.GetIndices();
            if (vertices == null || faces == null || vertices.Count == 0) continue;
            var indices = source.majorVersion >= 9 ? faces[0] : faces.SelectMany(f => f).ToList();
            if (indices.Any(i => i < 0 || i >= vertices.Count / 3)) throw new Exception("Mesh index outside vertex range");
            meshes[id] = new { vertices, indices };
        }
        ids.Add(id);
    }
    // Each original JT node is retained; only first LOD is used per shape.
    nodes.Add(new { id = node.id, type = node.elementData.GetType().Name, parent = node.parent?.id, children = node.child.Select(c => c.id),
        transform, color = material == null ? null : new []{material.r, material.g, material.b, material.a},
        properties = node.property.Where(e => e.Value is not DataSegment).ToDictionary(e => e.Key,e => e.Value?.ToString()),
        meshes = ids.Take(1) });
}
Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(args[1])));
File.WriteAllText(args[1], JsonSerializer.Serialize(new { source = args[0], version = source.Version,
    root = lsg.root.id, nodes, meshes, rawPmi, metadata = source.segments.Values.SelectMany(s => s.elements ?? new List<Element>()).Where(e => e.elementData is PropertyProxyMetaData).Select(e => new { e.objectID, properties = ((PropertyProxyMetaData)e.elementData).property.ToDictionary(p => p.Key, p => p.Value.Item2?.ToString()) }), kinematics = lsg.txKin?.devices, kinematicsSource = source.txKinModelingBuffer },
    new JsonSerializerOptions { IncludeFields = true }));
Console.WriteLine($"Imported JT {source.Version}: {nodes.Count} nodes, {meshes.Count} meshes -> {args[1]}");
Console.WriteLine(string.Join(", ", source.segments.Values.SelectMany(s => s.elements ?? new List<Element>()).GroupBy(e => e.elementData?.GetType().Name ?? "null").Select(g => $"{g.Key}:{g.Count()}")));
